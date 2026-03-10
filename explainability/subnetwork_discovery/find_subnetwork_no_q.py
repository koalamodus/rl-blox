from flax import nnx
from rl_blox.blox.function_approximator.mlp import MLP
import jax.numpy as jnp
import jax
import jax.random as jr
import optax
import gymnasium as gym
import pickle
import chex

# ---------------------------
# (1) Load trained network and replay buffer
# ---------------------------
env_name = "CartPole-v1"
env = gym.make(env_name)
seed = 42

# Recreate MLP with same architecture as training
hparams_model = dict(
    activation="relu",
    hidden_nodes=[128, 128],
)

q = MLP(env.observation_space.shape[0], int(env.action_space.n), rngs=nnx.Rngs(seed), **hparams_model)

# Load trained MLP parameters
with open("ddqn_model_ckpt.pkl", "rb") as f:
    mlp_state = pickle.load(f)
nnx.update(q, mlp_state)
print("Loaded trained MLP parameters successfully")

# Load replay buffer (for behavior cloning dataset)
with open("rb_ckpt.pkl", "rb") as f:
    rb = pickle.load(f)
states = jnp.array(rb.buffer['observation'])
actions = jnp.array(rb.buffer['action']).astype(jnp.int32)

chex.assert_equal_shape_prefix((states, actions), prefix_len=1)
print(f"Replay buffer length: {len(actions)}")


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

# ---------------------------
# (3) Behavior cloning loss for circuit discovery
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

# def soft_mask_sparsity(masks: dict) -> jnp.ndarray:
#     """Compute total sparsity penalty over all mask logits."""

#     hidden_terms = [
#         jnp.sum(jax.nn.sigmoid(param_mask.value))
#         for layer_masks in masks["hidden_layers"]
#         for param_mask in layer_masks.values()
#     ]

#     output_terms = [
#         jnp.sum(jax.nn.sigmoid(param_mask.value))
#         for param_mask in masks["output_layer"].values()
#     ]

#     all_terms = hidden_terms + output_terms

#     return jnp.sum(jnp.stack(all_terms))

# def q_diff_loss(q_values_diff):
#     # Loss = mean squared difference between original Q and masked Q
#     return jnp.mean((q_values_diff) ** 2)

# def subnetwork_loss(masked_net, q_values_diff, sparsity_lambda=1e-3):
#     q_diff_loss_val = q_diff_loss(q_values_diff)

#     sparsity_loss = sparsity_lambda * soft_mask_sparsity(masked_net.masks)
#     total_loss = q_diff_loss_val + sparsity_loss

#     # jax.debug.print("total_loss = {z}, q_diff_loss_val = {x}, sparsity_loss = {y}",
#     #                  x=q_diff_loss_val, y=sparsity_loss, z=total_loss)

#     metrics = {
#         'q_diff_loss': q_diff_loss_val,
#         'sparsity_loss': sparsity_loss
#     }
#     return total_loss, metrics

def q_diff_loss(masked_net: MaskedMLP, q_net: MLP, batch):
    """
    Loss is the mean squared difference between original Q and masked Q
    """
    q_masked = masked_net(batch['obs'])
    q_original = q_net(batch['obs'])
    q_values_diff = q_masked - jax.lax.stop_gradient(q_original)

    q_diff_loss_val = jnp.mean(q_values_diff ** 2)
    return q_diff_loss_val

def subnetwork_loss(masked_net: MaskedMLP, q_net: MLP, batch, sparsity_lambda=1e-3):
    """
    Compute the loss for a masked subnetwork compared to a frozen Q-network.

    Args:
        masked_net: MaskedMLP being optimized
        q_net: Pretrained frozen Q-network
        batch: Dict with keys 'obs' and 'act'
        sparsity_lambda: Weight for mask sparsity regularization
    """
    q_diff_loss_val = q_diff_loss(masked_net, q_net, batch)
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

batch_size = 128
num_steps = 50_000
lambda_start, lambda_end = 1e-6, 1e-6
warmup_steps = 3_000
threshold_eval = 0.5
lr = 3e-4

optimizer = nnx.Optimizer(masked_net, optax.adam(lr), wrt=nnx.Param)
# optimizer = nnx.Optimizer(masked_net.mask_logits, optax.adam(lr), wrt=nnx.Param)

# TODO: remove replay buffer, sample directly from state
def sample_batch(rb, batch_size=128, key=jr.PRNGKey(0)):
    N = rb.buffer['observation'].shape[0]
    idx = jr.randint(key, (batch_size,), minval=0, maxval=N)
    obs = jnp.array(rb.buffer['observation'])[idx]
    act = jnp.array(rb.buffer['action'])[idx].astype(jnp.int32)
    return {'obs': obs, 'act': act}

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
train_step = partial(nnx.jit, static_argnames=("gamma",))(train_step)

key = jr.PRNGKey(seed + 123)
for step in tqdm(range(1, num_steps + 1), desc="Training"):
    key, subkey = jr.split(key)
    batch = sample_batch(rb, batch_size, key=subkey)
    lam = sparsity_schedule(step, lambda_start, lambda_end, warmup_steps, num_steps)

    # q_values_diff = masked_net(batch['obs']) - q(batch['obs'])
    # # jax.debug.print("out_diff = {out_diff}",
    # #                  out_diff=q.output_layer.kernel.value - masked_net.masked_mlp.output_layer.kernel.value)
    # # q_values_diff = masked_net.masked_mlp(batch['obs']) - q(batch['obs'])
    # # q_diff_loss_val = jnp.mean((q_values_diff) ** 2)
    # # print(f"q_values_diff:{q_values_diff}")
    # # print(f"q_diff_loss_val:{q_diff_loss_val}")

    # loss_val, metrics = train_step(optimizer, masked_net, q_values_diff, lam)
    # # (loss_val, metrics), grads = nnx.value_and_grad(subnetwork_loss, has_aux=True)(masked_net, batch, lam)
    # # optimizer.update(masked_net, grads)

   
    loss_val, metrics = train_step(optimizer, masked_net, q, batch, lam)

    if step % 100 == 0:
        sparsity = sparsity_fraction(masked_net, threshold_eval)
        print(f"step={step} loss={loss_val:.6f} q_diff_loss={metrics['q_diff_loss']:.6f} "
              f"sparsity_loss={metrics['sparsity_loss']:.6f} sparsity@{threshold_eval}={sparsity:.3f}")
        
        # early stopping
        if sparsity < 0.5: #and metrics['q_diff_loss'] < 0.01:
            print("original q-net")
            print(nnx.state(q))
            # jax.debug.print("q_out = {q_out}, masknet_out = {masknet_out}",
            #         q_out=q.output_layer.kernel.value, masknet_out=masked_net.masked_mlp.output_layer.kernel.value)
            # jax.debug.print("mask = {mask}", mask=masked_net.masks)
            break

# ---------------------------
# (5) Extract subnetwork
# ---------------------------

import jax
from flax import nnx

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
print(f"pruned_weights: {pruned_weights}")

q_pruned = MLP(
    env.observation_space.shape[0],
    int(env.action_space.n),
    rngs=nnx.Rngs(seed),  # any RNG; will be overwritten by pruned weights
    **hparams_model
)

# For hidden layers: convert list of dicts to dict of dicts keyed by index (optional)
hidden_layers_dict = {i: layer for i, layer in enumerate(pruned_weights["hidden_layers"])}

# Merge with output layer
pruned_state = {
    "hidden_layers": hidden_layers_dict,
    "output_layer": pruned_weights["output_layer"]
}

# Update the new MLP with pruned parameters
nnx.update(q_pruned, pruned_state)

# create a pruned MLP for evaluation
# q_pruned = masked_net.pruned_mlp

# save pruned network
subnetwork_state = nnx.state(q_pruned)
print(f"subnetwork_state is {subnetwork_state}")


# with open("ddqn_subnet_ckpt.pkl", "wb") as f:
#     pickle.dump(subnetwork_state, f)
# print("Saved subnetwork checkpoint.")

# Show the subnetwork policy
eval_env = gym.make(env_name, render_mode="human")
obs, _ = eval_env.reset()

while True:
    action = int(jnp.argmax(q_pruned([obs])))
    next_obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, _ = eval_env.reset()
    else:
        obs = next_obs