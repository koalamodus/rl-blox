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
    """Masked wrapper for a pretrained MLP for circuit discovery."""
    
    def __init__(self, base_mlp: MLP, rngs: nnx.Rngs, init_bias_logit=0.9, init_std=0.01, mask_bias=False):
        self.base = base_mlp
        self.mask_bias = mask_bias
        graphdef, base_state = nnx.split(self.base)

        self.masks = {}

        def create_masks(layer_params):
            masks = {}
            for param_name, param_value in layer_params.items():
                if param_name == "kernel" or (mask_bias and param_name == "bias"):
                    shape = param_value.value.shape
                    key = rngs()
                    init = jax.nn.initializers.normal(init_std)(key, shape) + init_bias_logit
                    masks[param_name] = nnx.Param(init)
            return masks

        # hidden layers
        self.masks["hidden_layers"] = {
            layer_idx: create_masks(base_state["hidden_layers"][layer_idx])
            for layer_idx in base_state["hidden_layers"]
        }

        # output layer
        self.masks["output_layer"] = create_masks(base_state["output_layer"])

        self.masked_mlp = nnx.merge(graphdef, self._masked_state(), copy=True)
    
    def _masked_state(self):
        """Return a masked copy of base network parameters."""
        base_state = nnx.state(self.base)
        masked_state = {}

        def apply_masks(layer_params, mask_params):
            out = {}
            for param_name, param_value in layer_params.items():
                if param_name in mask_params:
                    mask_soft = jax.nn.sigmoid(mask_params[param_name].value)
                    mask_hard = (mask_soft >= 0.5).astype(param_value.value.dtype)
                    # Straight-through gradient: the mask is soft mask in backprop and hard mask in forward
                    mask = mask_soft + jax.lax.stop_gradient(mask_hard - mask_soft)

                    out[param_name] = jax.lax.stop_gradient(param_value.value) * nnx.Param(mask)
                else:
                    out[param_name] = jax.lax.stop_gradient(param_value)
            return out

        # hidden layers
        masked_state["hidden_layers"] = {
            layer_idx: apply_masks(
                base_state["hidden_layers"][layer_idx],
                self.masks["hidden_layers"][layer_idx],
            )
            for layer_idx in base_state["hidden_layers"]
        }

        # output layer
        masked_state["output_layer"] = apply_masks(
            base_state["output_layer"],
            self.masks["output_layer"],
        )

        return nnx.state(masked_state)

    def __call__(self, x):
        return self.masked_mlp(x)
    def hard_threshold_params(self, threshold=0.5):
        """Return pruned parameters with masks hard-thresholded."""
        graphdef, base_state = nnx.split(self.base)
        pruned_weights = {}

        def apply_threshold(layer_params, mask_params):
            out = {}
            for param_name, param_value in layer_params.items():
                if param_name in mask_params:
                    mask = (
                        jax.nn.sigmoid(mask_params[param_name].value) >= threshold
                    ).astype(param_value.value.dtype)
                    out[param_name] =  jax.lax.stop_gradient(param_value.value) * nnx.Param(mask)
                else:
                    out[param_name] = jax.lax.stop_gradient(param_value)
            return out

        # hidden layers
        pruned_weights["hidden_layers"] = {
            layer_idx: apply_threshold(
                base_state["hidden_layers"][layer_idx],
                self.masks["hidden_layers"][layer_idx],
            )
            for layer_idx in base_state["hidden_layers"]
        }

        # output layer
        pruned_weights["output_layer"] = apply_threshold(
            base_state["output_layer"],
            self.masks["output_layer"],
        )
        self.pruned_mlp = nnx.merge(graphdef, self._masked_state(), copy=True)
    
    def sparsity_fraction(self, threshold=0.5):
        """Return fraction of weights kept (kernels only by default)."""

        def count_masks(mask_params):
            masks = jnp.array([jax.nn.sigmoid(mask.value) for mask in mask_params.values()])
            hard = (masks >= threshold).astype(jnp.int32)
            kept = jnp.sum(hard)
            total = jnp.sum(jnp.array([mask.value.size for mask in mask_params.values()]))
            return kept, total

        # hidden layers
        hidden_kept_total = jnp.array([count_masks(layer_masks) for layer_masks in self.masks["hidden_layers"].values()])
        output_kept, output_total = count_masks(self.masks["output_layer"])

        # combine all counts
        all_kept = jnp.sum(hidden_kept_total[:, 0]) + output_kept
        all_total = jnp.sum(hidden_kept_total[:, 1]) + output_total

        return all_kept / jnp.maximum(1, all_total)

# Instantiate masked network
masked_net = MaskedMLP(q, nnx.Rngs(seed+2))
# nnx.display(masked_net)

# ---------------------------
# (3) Behavior cloning loss for circuit discovery
# ---------------------------
def soft_mask_sparsity(masks: dict) -> jnp.ndarray:
    """Compute total sparsity penalty over all mask logits."""

    # hidden layers
    hidden_sums = jnp.array([
        jnp.sum(jax.nn.sigmoid(param_mask.value))
        for layer_masks in masks.get("hidden_layers", {}).values()
        for param_mask in layer_masks.values()
    ])

    # output layer
    output_sums = jnp.array([
        jnp.sum(jax.nn.sigmoid(param_mask.value))
        for param_mask in masks.get("output_layer", {}).values()
    ])

    # combine all sums
    return jnp.sum(hidden_sums) + jnp.sum(output_sums)

def q_diff_loss(q_values_diff):
    # Loss = mean squared difference between original Q and masked Q
    return jnp.mean((q_values_diff) ** 2)

def subnetwork_loss(masked_net, q_values_diff, sparsity_lambda=1e-3):
    q_diff_loss_val = q_diff_loss(q_values_diff)

    sparsity_loss = sparsity_lambda * soft_mask_sparsity(masked_net.masks)
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

    q_values_diff = masked_net.masked_mlp(batch['obs']) - q(batch['obs'])

    loss_val, metrics = train_step(optimizer, masked_net, q_values_diff, lam)
    # (loss_val, metrics), grads = nnx.value_and_grad(subnetwork_loss, has_aux=True)(masked_net, batch, lam)
    # optimizer.update(masked_net, grads)

   

    if step % 1000 == 0:
        sparsity = masked_net.sparsity_fraction(threshold_eval)
        print(f"step={step} loss={loss_val:.6f} q_diff_loss={metrics['q_diff_loss']:.6f} "
              f"sparsity_loss={metrics['sparsity_loss']:.6f} sparsity@{threshold_eval}={sparsity:.3f}")
        
        # early stopping
        if sparsity < 0.1 and metrics['q_diff_loss'] < 0.01:
            break

# ---------------------------
# (5) Extract subnetwork
# ---------------------------
masked_net.hard_threshold_params(threshold_eval)

# create a pruned MLP for evaluation
q_pruned = masked_net.pruned_mlp
print(f"Pruned network ready, fraction of weights kept: {masked_net.sparsity_fraction(threshold_eval):.3f}")

# save pruned network
subnetwork_state = nnx.state(q_pruned)
print(f"subnetwork_state is {subnetwork_state}")


with open("ddqn_subnet_ckpt.pkl", "wb") as f:
    pickle.dump(subnetwork_state, f)
print("Saved subnetwork checkpoint.")

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