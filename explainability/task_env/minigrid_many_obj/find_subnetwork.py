import os
from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX
from find_many_objects_env import make_ocean_env

import jax
import flax.nnx as nnx
import jax.numpy as jnp
import jax.random as jr

seed = 49  # random seed for np and jax
key = jr.PRNGKey(seed)

# ---------------------------
# (1) Load trained network
# ---------------------------
import orbax.checkpoint as ocp
from rl_blox.blox.function_approximator.mlp import MLP

# Choose ckpt

experiment = "reefshield_random_medium"
randomize_env = True
seed_num = 0
file_name = "_minigrid_reefshield_random_medium_DDQN-UTS_1773787402.9864454_q_step_005000000_epoch_5000000"
model_hidden_nodes = [32, 32]

# experiment = "reefshield_fixed_medium"
# randomize_env = False
# seed_num = 0
# file_name = "_minigrid_reefshield_fixed_medium_DDQN-UTS_1773675569.892775_q_step_001000000_epoch_1000000"
# model_hidden_nodes = [32, 32]

# experiment = "XRL_MINIGRID_POLICY"
# randomize_env = False
# seed_num = 48
# file_name = "minigrid_reefshield_medium_DDQN-UTS_1773420833.4308155_q_step_001000000_epoch_1000000"
# model_hidden_nodes = [128, 128]

# Set up ckpt path
ckpt_path = os.path.expanduser(
    f"~/workspace/XRL/ocean_trained_model/{experiment}/UTS/seed_{seed_num}/{file_name}"
)

# Set up environment to get input/output shapes
OBJ_COLORS = COLOR_NAMES
# subtask = None
subtask = "red"
env_name = f"ocean_{subtask}_{experiment}"
env = make_ocean_env(subtask)

# Recreate MLP with same architecture as training
hparams_model = dict(
    n_features=env.observation_space.shape[0],
    n_outputs=int(env.action_space.n),
    activation="relu",
    hidden_nodes=model_hidden_nodes,
)

# Restore q net from checkpoint
abstract_mlp = MLP(rngs=nnx.Rngs(seed), **hparams_model)
graphdef, abstract_state = nnx.split(abstract_mlp)


checkpointer = ocp.StandardCheckpointer()
restored_model = checkpointer.restore(ckpt_path, abstract_state)
q = nnx.merge(graphdef, restored_model)

print("Loaded model checkpoint.")


def get_policy_from_q_net(q):

    def policy(obs):
        return int(jnp.argmax(q([obs])))

    return policy


policy = get_policy_from_q_net(q)

eval_num_episode = 20 if randomize_env else 1

from eval_helper import eval_policy
def evaluate_policy_on_task(policy, task=None, obj_colors=OBJ_COLORS, randomize=randomize_env, num_episode=eval_num_episode, render_mode=None, seed=42, verbose=True):
    if task == None:
        task = obj_colors
    else:
        task = task if isinstance(task, list) else [task]
    
    all_task_scores = {}  # store results for all colors

    for target_color in task:
        assert target_color in COLOR_NAMES

        eval_env = make_ocean_env(target_color, randomize, render_mode)
        task_score_info = eval_policy(target_color, eval_env, policy, verbose=verbose, num_episode=num_episode, seed=seed)
        eval_env.close()

        all_task_scores.update(task_score_info)
    return all_task_scores

print("evaluate original q network")
q_eval_scores = evaluate_policy_on_task(policy, seed=seed)


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

# ---------------------------
# (3) Subnetwork loss and sparsity computation
# ---------------------------
def hard_mask_sparsity(mask_logits: dict, threshold: float = 0.5) -> jnp.ndarray:
    """Compute fraction of weights masked (hard mask >= threshold) in a MaskedMLP."""

    masks = []

    # hidden layers
    for layer in mask_logits["hidden_layers"]:
        # apply sigmoid to logits to get soft mask
        masks.append(jax.nn.sigmoid(layer["kernel"].value))

    # output layer
    masks.append(jax.nn.sigmoid(mask_logits["output_layer"]["kernel"].value))

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
    q_original = jax.lax.stop_gradient(q_net(state))

    # # MSE and abs error loss
    q_values_diff = q_masked - q_original
    # q_diff_loss_val = jnp.mean(q_values_diff ** 2)
    q_diff_loss_val = jnp.mean(jnp.abs(q_values_diff))

    # debug metrics
    abs_avg_q_original = jnp.mean(jnp.abs(q_original))
    abs_avg_q_masked = jnp.mean(jnp.abs(q_masked))
    return q_diff_loss_val, (abs_avg_q_original, abs_avg_q_masked)

def subnetwork_loss(masked_net: MaskedMLP, q_net: MLP, state, sparsity_lambda=1e-3):
    """
    Compute the loss for a masked subnetwork compared to a frozen Q-network.

    Args:
        masked_net: MaskedMLP being optimized
        q_net: Pretrained frozen Q-network
        batch: Dict with keys 'obs' and 'act'
        sparsity_lambda: Weight for mask sparsity regularization
    """
    q_diff_loss_val, (abs_avg_q_original, abs_avg_q_masked) = q_diff_loss(masked_net, q_net, state)
    soft_sparsity = soft_mask_sparsity(masked_net.mask_logits)
    sparsity_loss = sparsity_lambda * soft_sparsity
    total_loss = q_diff_loss_val + sparsity_loss

    hard_sparsity = hard_mask_sparsity(masked_net.mask_logits)

    metrics = {
        'q_diff_loss': q_diff_loss_val,
        'soft_sparsity': soft_sparsity,
        'hard_sparsity': hard_sparsity,
        'sparsity_loss': sparsity_loss,
        'abs_avg_q_original': abs_avg_q_original,
        'abs_avg_q_masked': abs_avg_q_masked,
    }

    return total_loss, metrics

# ---------------------------
# (4) Extract subnetwork & Evaluate pruned network
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

def get_pruned_state(masked_net: MaskedMLP, threshold_eval: float = 0.5):
    """
    Update the q_pruned MLP parameters using hard-thresholded masks from masked_net.
    Returns the updated pruned_state dict.
    """
    pruned_weights = hard_threshold_params(masked_net, threshold_eval)
    # print(f"pruned_weights: {pruned_weights}")


    # For hidden layers: convert list of dicts to dict of dicts keyed by index (optional)
    hidden_layers_dict = {i: layer for i, layer in enumerate(pruned_weights["hidden_layers"])}

    # Merge with output layer
    pruned_state = {
        "hidden_layers": hidden_layers_dict,
        "output_layer": pruned_weights["output_layer"]
    }
    return pruned_state

# ---------------------------
# (5) Training loop
# ---------------------------
from functools import partial
from tqdm import tqdm
import optax
from rl_blox.logging.logger import AIMLogger

eval_steps = 10

hparams_algorithm = dict(
    batch_size=1024,
    total_timesteps=1_500,
    learning_rate=1e-4,
    learning_rate_start=1e-3,
    learning_rate_end=1e-6,
    learning_starts=0,
    learning_ends=1_000,
    lambda_start= 1e-8,
    lambda_end= 1e-8,
    threshold_eval=0.5,
)

logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="DDQN_pruning",
    hparams=hparams_model | hparams_algorithm,
)

logger.run.log_info(f"random seed: {seed}")
logger.run.log_info(f"number of episode in evaluation: {eval_num_episode}")

logger.run.log_info("evaluate original q network")
logger.run.log_info(f"{q_eval_scores}")

threshold_eval = hparams_algorithm.get("threshold_eval")

# Set learning rate, use either scheduler or a constant value
lr_constant = hparams_algorithm.pop("learning_rate")
lr_schedule_fn = optax.linear_schedule(
    init_value=hparams_algorithm.pop("learning_rate_start"), end_value=hparams_algorithm.pop("learning_rate_end"),
    transition_steps=hparams_algorithm.get("learning_ends"), transition_begin=hparams_algorithm.get("learning_starts")
)

# lr = lr_constant
lr = lr_schedule_fn

# Initialise optimiser
optimizer = nnx.Optimizer(
    masked_net, optax.adam(learning_rate=lr), wrt=nnx.Param
)

def sample_state(env, subtask=None, batch_size=128, key=jr.PRNGKey(seed)):
    # Get the observation space bounds
    low = jnp.array(env.observation_space.low)
    high = jnp.array(env.observation_space.high)

    # low and high are currently 0 and 255, but in env it is 0 and 6
    key, subkey = jr.split(key)

    # Sample a batch of states uniformly
    state = jax.random.randint(
        subkey,
        shape=(batch_size, low.shape[0]),
        minval=low,
        maxval=high,
    )

    obj_colors = OBJ_COLORS

    if subtask is None:
        # Sample all subtasks

        # Get allowed indices
        obj_color_indices = jnp.array([COLOR_TO_IDX[c] for c in OBJ_COLORS])

        # sample index into ObJ_COLORS
        key, subkey = jr.split(key)
        idx = jax.random.randint(subkey, shape=(), minval=0, maxval=len(obj_color_indices))

        context = obj_color_indices[idx]
        # jax.debug.print("context={c}", c=context)

    elif subtask in obj_colors:
        # Sample only the subtask
        context = COLOR_TO_IDX[subtask]

    else:
        raise RuntimeError(f"Unknown subtask: {subtask}")

    # Dezimal context 
    # # broadcast to batch
    # context_batch = jnp.full((batch_size,), context)
    # # Replace the last elements of every row in `state`
    # state = state.at[:, -1].set(context_batch)

    # One-hot context
    one_hot_length = 6
    one_hot_context = jnp.zeros(one_hot_length)
    one_hot_context = one_hot_context.at[context].set(1)  # JAX-friendly

    # Broadcast the one-hot context to the batch
    # Replace the last 6 elements of every row in `state`
    state = state.at[:, -one_hot_length:].set(one_hot_context)

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

q_pruned = MLP(rngs=nnx.Rngs(seed), **hparams_model)


# Train mask
for step in tqdm(range(1, hparams_algorithm.get("total_timesteps") + 1), desc="Training"):
    key, subkey = jr.split(key)
    state = sample_state(env, subtask, hparams_algorithm.get("batch_size"), key=subkey)
    lam = sparsity_schedule(step, hparams_algorithm.get("lambda_start"), hparams_algorithm.get("lambda_end"),
                            hparams_algorithm.get("learning_starts"), hparams_algorithm.get("total_timesteps"))

    loss_val, training_metrics = train_step(optimizer, masked_net, q, state, lam)

    if logger is not None:
        logger.record_stat(
            "q diff loss", training_metrics['q_diff_loss'], step=step + 1
        )
        logger.record_stat(
            "sparsity loss", training_metrics['sparsity_loss'], step=step + 1
        )
        logger.record_stat(
            "soft sparsity", training_metrics['soft_sparsity'], step=step + 1
        )
        logger.record_stat(
            "hard sparsity", training_metrics['hard_sparsity'], step=step + 1
        )
        logger.record_stat(
            "avg abs q value", training_metrics['abs_avg_q_original'], step=step + 1
        )
        logger.record_stat(
            "avg abs masked q value", training_metrics['abs_avg_q_masked'], step=step + 1
        )

    # extract and evaluate subnet
    if step % eval_steps == 0: # and training_metrics['hard_sparsity'] < 1.0:
        # key, subkey = jr.split(key)
        pruned_state = get_pruned_state(masked_net, threshold_eval)
        # Update the new MLP with pruned parameters
        nnx.update(q_pruned, pruned_state)
        
        # Evaluate the subnetwork policy
        subnet_policy = get_policy_from_q_net(q_pruned)
        all_task_scores = evaluate_policy_on_task(subnet_policy, seed=seed+1, verbose=False)

        # Log eval result
        for color, eval_metrics in all_task_scores.items():
            logger.record_stat(
                f"{color} avg return", eval_metrics["Avg Return"], step=step + 1
            )
            logger.record_stat(
                f"{color} success rate", eval_metrics["Success Rate"], step=step + 1
            )
            if q_eval_scores[color]["Avg Return"] != 0:
                logger.record_stat(
                    f"{color} avg return (relative to original q network)", eval_metrics["Avg Return"]/q_eval_scores[color]["Avg Return"], step=step + 1
                )
            if q_eval_scores[color]["Success Rate"] != 0:
                logger.record_stat(
                    f"{color} success rate (relative to original q network)s", eval_metrics["Success Rate"]/q_eval_scores[color]["Success Rate"], step=step + 1
                )

    if step % 1 == 0 and step > 10000:
        # early stopping
        if training_metrics['hard_sparsity'] < 0.4 and training_metrics['q_diff_loss'] < 1e-2:
            print(f"step={step} loss={loss_val:.6f} q_diff_loss={training_metrics['q_diff_loss']:.6f} "
              f"sparsity_loss={training_metrics['sparsity_loss']:.6f} sparsity@{threshold_eval}={training_metrics['hard_sparsity']:.6f}")
            
            logger.run.log_info(f"evaluate subnetwork {subtask}")
            logger.run.log_info(f"{all_task_scores}")

            break

# ---------------------------
# (6) Extract subnetwork & Evaluate pruned network
# ---------------------------

print(f"Pruned network ready, fraction of weights kept: {hard_mask_sparsity(masked_net.mask_logits, threshold_eval):.3f}")

q_pruned = MLP(rngs=nnx.Rngs(seed), **hparams_model)

pruned_state = get_pruned_state(masked_net, threshold_eval)
# Update the new MLP with pruned parameters
nnx.update(q_pruned, pruned_state)

# save pruned network
subnetwork_state = nnx.state(q_pruned)
# print(f"subnetwork_state is {subnetwork_state}")


# Save subnetwork
checkpointer = ocp.StandardCheckpointer()

step = int(file_name.split("_")[-3])
ckpt_save_path = os.path.expanduser(
        f"~/workspace/XRL/ocean_subnet/{experiment}/UTS/seed_{seed_num}/step_{step}/subnetwork_{subtask}"
    )

checkpointer.save(
    ckpt_save_path,
    subnetwork_state
)
print(f"Saved {subtask} subnetwork checkpoint.")

# Evaluate the subnetwork policy
subnet_policy = get_policy_from_q_net(q_pruned)
# _ = evaluate_policy_on_task(subnet_policy, task=subtask, seed=seed+1)
subnet_eval_scores = evaluate_policy_on_task(subnet_policy, seed=seed+1)
logger.run.log_info(f"evaluate pruned subnetwork {subtask}")
logger.run.log_info(f"{subnet_eval_scores}")

# For comparison
print("evaluate original q network")
_ = evaluate_policy_on_task(policy, seed=seed+2)

# ---------------------------
# Run interactive demo
# ---------------------------
import random

task = random.choice(OBJ_COLORS)
sum_reward = 0.0
eval_env = make_ocean_env(task, randomize=randomize_env, render_mode="human")
obs, _ = eval_env.reset()
while True:
    action = subnet_policy(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    sum_reward += reward
    if terminated or truncated:
        print(f"{task} reward: {reward}")
        print(f"{task} return: {sum_reward}")
        eval_env.close()
        task = random.choice(OBJ_COLORS)
        sum_reward = 0.0
        eval_env = make_ocean_env(task, randomize=randomize_env, render_mode="human")
        obs, _ = eval_env.reset()