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

    def __init__(self, base_mlp: MLP, rngs: nnx.Rngs, init_bias_logit=4.0, init_std=0.01, mask_bias=False):
        self.base = base_mlp
        self.mask_bias = mask_bias
        base_state = nnx.state(self.base)

        # Initialize mask logits for all kernels (and optionally biases)
        self.masks = {}
        for layer_name, layer_params in base_state.items():
            self.masks[layer_name] = {}
            for param_name, param_value in layer_params.items():
                if param_name == 'kernel' or (mask_bias and param_name == 'bias'):
                    shape = param_value.value.shape
                    key = rngs()
                    init = jax.nn.initializers.normal(init_std)(key, shape) + init_bias_logit
                    self.masks[layer_name][param_name] = nnx.Param(init)

    def _masked_state(self):
        """Return a masked copy of base network parameters."""
        base_state = nnx.state(self.base)
        masked_state = {}
        for layer_name, layer_params in base_state.items():
            masked_state[layer_name] = {}
            for param_name, param_value in layer_params.items():
                if param_name in self.masks[layer_name]:
                    mask = jax.nn.sigmoid(self.masks[layer_name][param_name].value)
                    masked_state[layer_name][param_name] = nnx.Param(param_value.value * mask)
                else:
                    masked_state[layer_name][param_name] = param_value
        return nnx.state(masked_state)

    def __call__(self, x):
        nnx.update(self.base, self._masked_state())
        return self.base(x)

    def hard_threshold_params(self, threshold=0.5):
        """Return pruned parameters with masks hard-thresholded."""
        base_state = nnx.state(self.base)
        pruned_weights = {}
        for layer_name, layer_params in base_state.items():
            pruned_weights[layer_name] = {}
            for param_name, param_value in layer_params.items():
                if param_name in self.masks[layer_name]:
                    mask = (jax.nn.sigmoid(self.masks[layer_name][param_name].value) >= threshold).astype(param_value.value.dtype)
                    pruned_weights[layer_name][param_name] = nnx.Param(param_value.value * mask)
                else:
                    pruned_weights[layer_name][param_name] = param_value
        return pruned_weights

    def sparsity_fraction(self, threshold=0.5):
        """Return fraction of weights kept (kernels only by default)."""
        total, kept = 0, 0
        for layer_name, layer_params in self.masks.items():
            for param_name, mask_param in layer_params.items():
                mask = jax.nn.sigmoid(mask_param.value)
                hard = (mask >= threshold).astype(jnp.int32)
                total += hard.size
                kept += int(hard.sum())
        return kept / max(1, total)

# Instantiate masked network
masked_net = MaskedMLP(q, nnx.Rngs(seed+2))

# ---------------------------
# (3) Behavior cloning loss for circuit discovery
# ---------------------------
def mask_penalty(masks: dict) -> jnp.ndarray:
    """Compute total sparsity penalty over all mask logits."""
    total = 0.0
    for layer in masks.values():  # e.g., hidden_layers, output_layer
        if isinstance(layer, dict):
            for param in layer.values():  # kernel / bias
                total += jnp.sum(jax.nn.sigmoid(param.value))
        else:
            total += jnp.sum(jax.nn.sigmoid(layer.value))
    return total

def bc_loss(masked_net, batch, sparsity_lambda=1e-3):
    """
    Behavior cloning loss with sparsity regularization.

    Args:
        masked_net: MaskedMLP instance
        batch: dict with 'obs' and 'act' arrays
        lam: sparsity regularization coefficient
    """
    # Forward pass
    logits = masked_net(batch['obs'])  # shape [batch, n_actions]

    # Action log-probs
    log_probs = jax.nn.log_softmax(logits)
    act_indices = batch['act'][:, None]
    selected_log_probs = jnp.take_along_axis(log_probs, act_indices, axis=1).squeeze(1)

    bc_loss_val = -jnp.mean(selected_log_probs)
    sparsity_penalty = sparsity_lambda * mask_penalty(masked_net.masks)
    total_loss = bc_loss_val + sparsity_penalty

    metrics = {
        'bc_loss': bc_loss_val,
        'sparsity_penalty': sparsity_penalty
    }

    return total_loss, metrics

# ---------------------------
# (4) Training loop
# ---------------------------
from tqdm import tqdm

batch_size = 128
num_steps = 50_000
lambda_start, lambda_end = 1e-6, 1e-2
warmup_steps = 3_000
threshold_eval = 0.5
lr = 3e-4

optimizer = nnx.Optimizer(masked_net, optax.adam(lr), wrt=nnx.Param)

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

key = jr.PRNGKey(seed + 123)
for step in tqdm(range(1, num_steps + 1), desc="Training"):
    key, sk = jr.split(key)
    batch = sample_batch(rb, batch_size, key=sk)
    lam = sparsity_schedule(step, lambda_start, lambda_end, warmup_steps, num_steps)
    (loss_val, metrics), grads = nnx.value_and_grad(bc_loss, has_aux=True)(masked_net, batch, lam)
    optimizer.update(masked_net, grads)

    if step % 1000 == 0:
        sparsity = masked_net.sparsity_fraction(threshold_eval)
        print(f"step={step} loss={loss_val:.6f} bc_loss={metrics['bc_loss']:.6f} "
              f"sparsity_penalty={metrics['sparsity_penalty']:.6f} sparsity@{threshold_eval}={sparsity:.3f}")

# ---------------------------
# (5) Extract circuit
# ---------------------------
pruned_params = masked_net.hard_threshold_params(threshold_eval)

# create a pruned MLP for evaluation
q_pruned = MLP(env.observation_space.shape[0], int(env.action_space.n), rngs=nnx.Rngs(seed+999), **hparams_model)
nnx.update(q_pruned, pruned_params)
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