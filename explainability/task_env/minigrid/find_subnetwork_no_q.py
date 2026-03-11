import os
from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX
from minigrid_envs import make_ocean_env

import jax
import flax.nnx as nnx
import jax.numpy as jnp

seed = 10  # random seed for np and jax
key = jax.random.PRNGKey(seed)
# ---------------------------
# (1) Load trained network
# ---------------------------

import orbax.checkpoint as ocp
from rl_blox.blox.function_approximator.mlp import MLP

# Used to map colors to integers
# COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
subtask = "red" # "red", "green", "blue", "purple", "yellow", "grey"
env = make_ocean_env(color = subtask)

# Recreate MLP with same architecture as training
hparams_model = dict(
    n_features=env.observation_space.shape[0],
    n_outputs=int(env.action_space.n),
    activation="relu",
    hidden_nodes=[512, 512],
    rngs=nnx.Rngs(seed)
)

abstract_mlp = MLP(**hparams_model)
graphdef, abstract_state = nnx.split(abstract_mlp)

def get_policy_from_q_net(q):

    def policy(obs):
        return int(jnp.argmax(q([obs])))

    return policy



train_mask = True
save_final_mask = train_mask
load_final_mask = not train_mask


benchmark, grid_size = "oceans", "medium"
model_seed = 3
ckpt_name = "minigrid_oceans_medium_DDQN-UTS_1769513294.4912446_q_step_001930001_epoch_502057"
ckpt_path = os.path.expanduser(
    f"~/workspace/XRL/ocean_trained_model/{benchmark}_{grid_size}/uts/seed_{model_seed}/{ckpt_name}"
)
print(ckpt_path)

# restore q net from checkpoint
full_path = os.path.abspath(ckpt_path)
print(f"ckpts: {full_path}")
checkpointer = ocp.StandardCheckpointer()
restored_model = checkpointer.restore(full_path, abstract_state)
q = nnx.merge(graphdef, restored_model)

policy = get_policy_from_q_net(q)


from eval_helper import eval_policy
def evaluate_policy_on_task(policy, task="all", render_mode = "human"):
    if task == "all":
        task = COLOR_NAMES
    else:
        task = task if isinstance(task, list) else [task]
    
    for color in task:
        assert color in COLOR_NAMES

        eval_env = make_ocean_env(color, render_mode)
        _ = eval_policy(color, eval_env, policy, verbose=True, num_episode=1)
        eval_env.close()

print("evaluate original q network")
evaluate_policy_on_task(policy, render_mode="None")

q_copy = nnx.clone(q)
# # evaluate_policy(env_name, q_copy)
# print("evaluate q_copy network")
# policy_copy = get_policy_from_q_net(q_copy)
# evaluate_policy_on_task(policy_copy, render_mode="None")

# ---------------------------
# (2) Define masked network
# ---------------------------
class MaskedMLP(nnx.Module):
    """Masked wrapper around frozen MLP weights (no base stored as parameters)."""

    def __init__(self, frozen_params, rngs: nnx.Rngs, init_bias_logit=0.1, init_std=0.01):
        """frozen_params: dict containing MLP weights (kernel, bias)"""
        self.frozen_params = frozen_params

        self.mask_logits = {}
        key = rngs()

        def init_mask(shape):
            subkey = rngs()
            init = jax.nn.initializers.normal(init_std)(subkey, shape) + init_bias_logit
            return nnx.Param(init)

        # hidden layers
        self.mask_logits["hidden_layers"] = []
        for layer_params in frozen_params["hidden_layers"]:
            self.mask_logits["hidden_layers"].append({
                "kernel": init_mask(layer_params["kernel"].shape)
            })

        # output layer
        self.mask_logits["output_layer"] = {
            "kernel": init_mask(frozen_params["output_layer"]["kernel"].shape)
        }

    def _mask(self, logits):
        mask_soft = jax.nn.sigmoid(logits)
        mask_hard = (mask_soft >= 0.5).astype(mask_soft.dtype)
        return mask_soft + jax.lax.stop_gradient(mask_hard - mask_soft)

    def __call__(self, x):
        # hidden layers
        for i, layer_params in enumerate(self.frozen_params["hidden_layers"]):
            mask = self._mask(self.mask_logits["hidden_layers"][i]["kernel"].value)
            w = jax.lax.stop_gradient(layer_params["kernel"]) * mask
            b = jax.lax.stop_gradient(layer_params["bias"])
            x = nnx.relu(x @ w + b)  # assuming relu activation

        # output layer
        mask = self._mask(self.mask_logits["output_layer"]["kernel"].value)
        w = jax.lax.stop_gradient(self.frozen_params["output_layer"]["kernel"]) * mask
        b = jax.lax.stop_gradient(self.frozen_params["output_layer"]["bias"])
        return x @ w + b
    
# Instantiate masked network
frozen_params = {
    "hidden_layers": [{"kernel": layer.kernel.value, "bias": layer.bias.value} 
                      for layer in q.hidden_layers],
    "output_layer": {"kernel": q.output_layer.kernel.value, "bias": q.output_layer.bias.value}
}

masked_net = MaskedMLP(frozen_params, nnx.Rngs(seed+2))
# nnx.display(masked_net)
masked_net_copy = nnx.clone(masked_net)

# ---------------------------
# (3) Subnetwork loss and sparsity computation
# ---------------------------
def sparsity_fraction(masked_net: MaskedMLP, threshold: float = 0.5) -> jnp.ndarray:
    """Compute fraction of weights masked (hard mask >= threshold) in a MaskedMLP."""

    masks = []

    # hidden layers
    for layer in masked_net.mask_logits["hidden_layers"]:
        # apply sigmoid to logits to get soft mask
        masks.append(jax.nn.sigmoid(layer["kernel"].value))

    # output layer
    masks.append(jax.nn.sigmoid(masked_net.mask_logits["output_layer"]["kernel"].value))

    # flatten and concatenate all masks
    masks_flat = jnp.concatenate([m.flatten() for m in masks])

    # hard mask based on threshold
    hard_mask = masks_flat >= threshold

    # return fraction of active weights
    return jnp.mean(hard_mask)

def soft_mask_sparsity(masks: dict) -> jnp.ndarray:
    """Compute total sparsity penalty over all mask logits."""

    # hidden layers
    hidden_sums = jnp.array([
        jnp.sum(jax.nn.sigmoid(param_mask.value))
        for layer_masks in masks["hidden_layers"]
        for param_mask in layer_masks.values()
    ])

    # output layer
    output_sums = jnp.array([
        jnp.sum(jax.nn.sigmoid(param_mask.value))
        for param_mask in masks["output_layer"].values()
    ])

    # combine all sums
    return jnp.sum(hidden_sums) + jnp.sum(output_sums)

def q_diff_loss(masked_net: MaskedMLP, q_net: MLP, state):
    """
    Loss is the mean squared difference between original Q and masked Q
    """
    q_masked = masked_net(state)
    q_original = q_net(state)
    q_values_diff = q_masked - jax.lax.stop_gradient(q_original)

    q_diff_loss_val = jnp.mean(q_values_diff ** 2)
    return q_diff_loss_val

def subnetwork_loss(masked_net: MaskedMLP, q_net: MLP, state, sparsity_lambda=1e-3):
    """
    Compute the loss for a masked subnetwork compared to a frozen Q-network.

    Args:
        masked_net: MaskedMLP being optimized
        q_net: Pretrained frozen Q-network
        batch: Dict with keys 'obs' and 'act'
        sparsity_lambda: Weight for mask sparsity regularization
    """
    q_diff_loss_val = q_diff_loss(masked_net, q_net, state)
    sparsity_loss = sparsity_lambda * soft_mask_sparsity(masked_net.mask_logits)
    total_loss = q_diff_loss_val + sparsity_loss

    metrics = {
        'q_diff_loss': q_diff_loss_val,
        'sparsity_loss': sparsity_loss
    }

    return total_loss, metrics

# ---------------------------
# (4) Training loop
# ---------------------------
from functools import partial
from tqdm import tqdm
import optax
import jax.random as jr
from debug_helper import compare_q_states, compare_mask_states

batch_size = 128
num_steps = 50_000
lambda_start, lambda_end = 1e-8, 1e-8
warmup_steps = 3_000
threshold_eval = 0.5
lr = 1e-4

optimizer = nnx.Optimizer(masked_net, optax.adam(lr), wrt=nnx.Param)

def sample_state(env, subtask, batch_size, key):
    # Get the observation space bounds
    low = jnp.array(env.observation_space.low)
    high = jnp.array(env.observation_space.high)

    # low and high are currently 0 and 255, but in env it is 0 and 6
    key, subkey = jr.split(key)

    # Sample a batch of states uniformly
    state = jax.random.randint(
        subkey,
        shape=(batch_size, low.shape[0]),
        minval=0,
        maxval=7,
    )
    # jax.debug.print("state={state}", state=state)

    # Get the context index
    try:
        context = COLOR_TO_IDX[subtask]
    except KeyError:
        raise RuntimeError(f"Unknown subtask: {subtask}")

    # One-hot context
    one_hot_length = 6
    one_hot_context = jnp.zeros(one_hot_length)
    one_hot_context = one_hot_context.at[context].set(1)  # JAX-friendly

    # Broadcast the one-hot context to the batch
    # Replace the last 6 elements of every row in `state`
    state = state.at[:, -one_hot_length:].set(one_hot_context)
    # jax.debug.print("state={state}", state=state)
    # jax.debug.print("-------------")

    return state

def sparsity_schedule(step, lambda_start, lambda_end, warmup_steps, total_steps):
    if step < warmup_steps:
        return lambda_start
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return float(lambda_start + progress * (lambda_end - lambda_start))

def train_step_with_loss(
    loss, optimizer: nnx.Optimizer, net: nnx.Module, *args, **kwargs
) -> tuple[float, float]:
    """Performs a single training step to optimize a network."""
    grad_fn = nnx.value_and_grad(loss, argnums=0, has_aux=True)
    value, grad = grad_fn(net, *args, **kwargs)
    optimizer.update(net, grad)
    return value

train_step = partial(train_step_with_loss, subnetwork_loss)
train_step = partial(nnx.jit, static_argnames=("lam",))(train_step)

key = jr.PRNGKey(seed + 123)
for step in tqdm(range(1, num_steps + 1), desc="Training"):
    key, subkey = jr.split(key)
    state = sample_state(env, subtask, batch_size, key=subkey)
    lam = sparsity_schedule(step, lambda_start, lambda_end, warmup_steps, num_steps)
   
    loss_val, metrics = train_step(optimizer, masked_net, q, state, lam)

    if step % 100 == 0:
        sparsity = sparsity_fraction(masked_net, threshold_eval)
        print(f"step={step} loss={loss_val:.6f} q_diff_loss={metrics['q_diff_loss']:.6f} "
              f"sparsity_loss={metrics['sparsity_loss']:.6f} sparsity@{threshold_eval}={sparsity:.6f}")
        
        # early stopping
        if sparsity < 0.85 and metrics['q_diff_loss'] < 1e-5:
            # jax.debug.print("-----------------q_net check-----------------")
            compare_q_states(q, q_copy)
            # jax.debug.print("-----------------masked_net check-----------------")
            compare_mask_states(masked_net, masked_net_copy)
            break

        # if sparsity == 1.0 and metrics['q_diff_loss'] != 0.0:
        #     jax.debug.print("q_diff_loss={l}", l=metrics['q_diff_loss'])

# ---------------------------
# (5) Extract subnetwork
# ---------------------------
def hard_threshold_params(masked_net, threshold=0.5):
    """
    Return a dict of frozen weights with masks hard-thresholded.
    Works for any MaskedMLP instance (or similar structure).
    """
    pruned_weights = {}

    def apply_threshold(layer_params, mask_params):
        out = {}
        for name, value in layer_params.items():
            if name in mask_params:
                mask = (jax.nn.sigmoid(mask_params[name].value) >= threshold).astype(value.dtype)
                out[name] = jax.lax.stop_gradient(value * mask)
            else:
                out[name] = jax.lax.stop_gradient(value)
        return out

    pruned_weights["hidden_layers"] = [
        apply_threshold(lp, mp)
        for lp, mp in zip(masked_net.frozen_params["hidden_layers"], masked_net.mask_logits["hidden_layers"])
    ]

    pruned_weights["output_layer"] = apply_threshold(
        masked_net.frozen_params["output_layer"], masked_net.mask_logits["output_layer"]
    )

    return pruned_weights

print(f"Pruned network ready, fraction of weights kept: {sparsity_fraction(masked_net, threshold_eval):.3f}")
pruned_weights = hard_threshold_params(masked_net, threshold_eval)
# print(f"pruned_weights: {pruned_weights}")

q_pruned = MLP(**hparams_model)

# For hidden layers: convert list of dicts to dict of dicts keyed by index (optional)
hidden_layers_dict = {i: layer for i, layer in enumerate(pruned_weights["hidden_layers"])}

# Merge with output layer
pruned_state = {
    "hidden_layers": hidden_layers_dict,
    "output_layer": pruned_weights["output_layer"]
}

# Update the new MLP with pruned parameters
nnx.update(q_pruned, pruned_state)

# save pruned network
subnetwork_state = nnx.state(q_pruned)
# print(f"subnetwork_state is {subnetwork_state}")


# jax.debug.print("-----------------q_pruned check-----------------")
compare_q_states(q, q_pruned, pruned=True)

# with open("ddqn_subnet_ckpt.pkl", "wb") as f:
#     pickle.dump(subnetwork_state, f)
# print("Saved subnetwork checkpoint.")

# Evaluate the subnetwork policy
subnet_policy = get_policy_from_q_net(q_pruned)
evaluate_policy_on_task(subnet_policy, task=subtask, render_mode="human")