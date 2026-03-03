import os
from minigrid.core.constants import COLOR_NAMES
from minigrid_envs import make_ocean_env

import jax

seed = 10  # random seed for np and jax
key = jax.random.PRNGKey(seed)
# ---------------------------
# (1) Load trained network
# ---------------------------

import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp
from rl_blox.blox.function_approximator.mlp import MLP

def get_mlp_with_state(env, state):
    abstract_mlp = MLP(
        int(env.observation_space.shape[0]),
        int(env.action_space.n),
        [512, 512],
        "relu",
        nnx.Rngs(0),
    )

    graphdef, abstract_state = nnx.split(abstract_mlp)
    mlp = nnx.merge(graphdef, state)

    return mlp

def get_q_net_from_checkpoint(path, env):
    full_path = os.path.abspath(path)
    print(f"ckpts: {full_path}")
    abstract_mlp = MLP(
        int(env.observation_space.shape[0]),
        int(env.action_space.n),
        [512, 512],
        "relu",
        nnx.Rngs(0),
    )

    graphdef, abstract_state = nnx.split(abstract_mlp)
    checkpointer = ocp.StandardCheckpointer()
    restored_model = checkpointer.restore(full_path, abstract_state)

    q = nnx.merge(graphdef, restored_model)

    return q

def get_policy_from_q_net(q):

    def policy(obs):
        return int(jnp.argmax(q([obs])))

    return policy

# Used to map colors to integers
# COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
subtask = "red" # "red", "green", "blue", "purple", "yellow", "grey"

env = make_ocean_env(color = subtask, render_mode = "human")

benchmark, grid_size = "oceans", "medium"
model_seed = 3
ckpt_name = "minigrid_oceans_medium_DDQN-UTS_1769513294.4912446_q_step_001930001_epoch_502057"
ckpt_path = os.path.expanduser(
    f"~/workspace/XRL/ocean_trained_model/{benchmark}_{grid_size}/uts/seed_{model_seed}/{ckpt_name}"
)
print(ckpt_path)

q = get_q_net_from_checkpoint(ckpt_path, env)
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

# ---------------------------
# (2) Define masked network
# ---------------------------

import jax
import jax.numpy as jnp

def get_kernel_shapes(mlp):
    kernel_shapes = {}

    # Hidden layers
    for i, layer in enumerate(mlp.hidden_layers):
        kernel_shapes[f"hidden_layer_{i}"] = layer.kernel.value.shape

    # Output layer
    kernel_shapes["output_layer"] = mlp.output_layer.kernel.value.shape

    return kernel_shapes


class MaskNet(nnx.Module):
    def __init__(self, shapes, init_value=0.9):
        # For each layer we store a Param
        for key, shape in shapes.items():
            setattr(self, key, nnx.Param(jnp.full(shape, init_value)))


def hard_concrete_sample(logits, tau, key, hard_threshold = 0.5):
    u1, u2 = jax.random.uniform(key, (2,))
    # print(f"u1, u2={u1, u2}")
    noise = -jnp.log(jnp.log(u1)/jnp.log(u2))
    # noise = -jnp.log(jnp.log(u1)) - jnp.log(jnp.log(u2))
    s = jax.nn.sigmoid((logits - noise) / tau)
    # print(f"noise={noise}, logits={logits}")
    
    hard = (s > hard_threshold).astype(s.dtype)
    
    # Straight-through estimator:
    # forward = hard, backward = gradient of s
    b = s + jax.lax.stop_gradient(hard - s)
    # print(f"b={b}, s={s}")

    return b


def apply_mask_to_layer(layer_key, mask_param, logit_val, tau, key):
    """
    Apply a hard‑concrete mask to the weight kernel of a layer using `logit_val`.
    """
    new_params = {}

    # TODO: check details
    for pname, pval in mask_param.items():
        if pname == "kernel":
            key, subkey = jax.random.split(key)

            # sample concrete mask from logits
            mask = hard_concrete_sample(logit_val, tau, subkey)

            # apply it
            new_params[pname] = pval.value * mask
            
            if len(pval.value) != len(mask):
                print(f"len(pval.value):{len(pval.value)}, len(mask):{len(mask)}")

        elif pname == "bias":
            new_params[pname] = pval.value

        else:
            raise ValueError(f"Unexpected param name in layer {layer_key}: {pname}")

    return new_params

def apply_mask_to_q(q, mask_model, tau, key):
    # Get the full state for network q
    state = nnx.state(q)  # A nested State mapping of layers and Param states
    new_state = {}

    for layer_name, q_param in state.items():
        if layer_name == "hidden_layers":
            new_state[layer_name] = {}

            for idx, layer_params in q_param.items():
                layer_key = f"hidden_layer_{idx}"

                # Extract the mask for this layer from mask_model
                mask_logits = getattr(mask_model, layer_key).value

                # Apply the mask to this layer’s q_param
                key, subkey = jax.random.split(key)
                new_state[layer_name][idx] = apply_mask_to_layer(
                    layer_key, layer_params, mask_logits, tau, subkey
                )

        elif layer_name == "output_layer":
            # Output layer mask
            layer_key = "output_layer"
            mask_logits = getattr(mask_model, layer_key).value

            key, subkey = jax.random.split(key)
            new_state[layer_name] = apply_mask_to_layer(
                layer_key, q_param, mask_logits, tau, subkey
            )
        else:
            raise ValueError(f"Unexpected layer: {layer_name}")

    masked_q_net = get_mlp_with_state(env, nnx.state(new_state))
    return masked_q_net


# 1. Get shapes
shapes = get_kernel_shapes(q)
print(f"kernel shapes:")
for name, shape in shapes.items():
    print(name, shape)

# 2. Create trainable mask logits
mask_model = MaskNet(shapes, init_value=0.9)

# 3. Use mask during forward pass
key, subkey = jax.random.split(key)
masked_q_net = apply_mask_to_q(q, mask_model, tau=0.5, key=subkey)
state_masked_q = nnx.state(masked_q_net)

# # 4. Masked q net
# # hard masked in forward pass, soft masked in back prop
# masked_policy = get_policy_from_q_net(masked_q_net)
# evaluate_policy_on_task(masked_policy, render_mode=None)


# ---------- L0 regularization over mask logits ----------
def l0_regularization(mask_model, lmbda):
    state = nnx.state(mask_model)

    all_logits = jnp.concatenate(
        [v.value.ravel() for v in state.values()]
    )
    all_logits_sizs = all_logits.size
    return lmbda * jnp.sum(jax.nn.sigmoid(all_logits)) / all_logits_sizs

# ---------- Simple rollout + replay buffer utilities ----------

import warnings
from tqdm import tqdm
from rl_blox.blox.replay_buffer import ReplayBuffer

def collect_task_data(rb, num_steps, q, task):
    env = make_ocean_env(color = task, render_mode = "None")
    obs, _ = env.reset()
    
    # TODO: try to sample state directly
    obs, _ = env.reset()
    for _ in tqdm(range(num_steps), desc="Experience collection"):
        #action = env.action_space.sample()
        action = int(jnp.argmax(q([obs])))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rb.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            termination=terminated or truncated,
        )
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

# Initialise the replay buffer
hparams_algorithm = dict(
    buffer_size=50_000,
    total_timesteps=50_000,
    seed=seed,
)

if hparams_algorithm["buffer_size"] > hparams_algorithm["total_timesteps"]:
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    warnings.warn(
        f"{YELLOW}ReplayBuffer size ({hparams_algorithm['buffer_size']}) is larger than "
        f"the total experience collected ({hparams_algorithm['total_timesteps']}). "
        f"ReplayBuffer will not be full.{RESET}",
        UserWarning,
    )

rb = ReplayBuffer(hparams_algorithm.pop("buffer_size"), discrete_actions=True)
num_steps = hparams_algorithm.pop("total_timesteps")

# Load existing replay buffer
import pickle

rb_file = f"rb_{subtask}.pkl"
rb_path = os.path.expanduser(
    f"~/workspace/XRL/ocean_subnet/{benchmark}_{grid_size}/uts/seed_{model_seed}/{ckpt_name}/{rb_file}"
)
os.makedirs(os.path.dirname(rb_path), exist_ok=True)

load_rb = False
if load_rb:
    with open(rb_path, "rb") as f:
        rb = pickle.load(f)
        print("Number of transitions:", len(rb))

if len(rb) < rb.buffer_size :
    # Collect experience with trained policy
    collect_task_data(rb, num_steps, q, task=subtask)


# Save experience in replay buffer
save_rb = True
if save_rb:
    import pickle
    with open(rb_path, "wb") as f:
        pickle.dump(rb, f)
    print(f"Saved replay buffer for task {subtask}.")

# # Test sample
# import numpy as np
# rng = np.random.default_rng(seed)

# num_train_steps=10000
# batch_size=64
# for step in range(num_train_steps):
#     batch = rb.sample_batch(batch_size, rng)
#     print(f"batch={batch}")

# ---------- Loss over mask logits (Q fixed, mask trainable) ----------

def print_values(q_values, q_sa, action, masked_q_values, masked_q_sa, q_diff_loss, l0, total_loss, index):
    i = index
    print(f"i={i}\n q_values={q_values[i]}, q_sa={q_sa[i]}, action={action[i]}\n masked_q_values={masked_q_values[i]}, masked_q_sa={masked_q_sa[i]}\n q_diff_loss={q_diff_loss}, l0={l0}, total_loss={total_loss}\n")
    # print(f"i={i}\n q_values={q_values[i]}, q_sa={q_sa[i]}, action={action[i]}")
    # jax.debug.print(f"masked_q_values={masked_q_values[i]}, masked_q_sa={masked_q_sa[i]}\n q_diff_loss={q_diff_loss}, l0={l0}, total_loss={total_loss}\n")

def mask_loss_fn(mask_params, q_sa, masked_q_sa, lmbda):

    # Loss = mean squared difference between original Q and masked Q
    q_diff_loss = jnp.mean((q_sa - masked_q_sa) ** 2)

    # L0 regularization on mask logits
    l0 = l0_regularization(mask_params, lmbda)

    total_loss = q_diff_loss + l0

    # index = jax.random.randint(subkey, (), 0, q_values.shape[0])
    # print_values(q_values, q_sa, batch["action"], masked_q_values, masked_q_sa, q_diff_loss, l0, total_loss, index)

    return total_loss, (q_diff_loss, l0)


mask_loss_grad_fn = jax.value_and_grad(mask_loss_fn, argnums=0, has_aux=True)

# ---------- Optimization loop for mask ----------
import optax

def q_sa_from_network(network, observations, actions):
    """
    Computes Q(s, a) for given observations and actions
    using the provided network.
    
    Args:
        network: Callable that maps observations -> Q-values
        observations: Batch of observations
        actions: Batch of actions (shape: [batch_size])
        
    Returns:
        Q(s, a) values (shape: [batch_size])
    """
    q_values = network(observations)
    q_sa = jnp.take_along_axis(
            q_values, actions[..., None], axis=1
        ).squeeze(-1)
    return q_sa

def optimize_mask_with_rb(q, mask_model,
                          rb,
                          key,
                          num_train_steps=5000,
                          batch_size=64,
                          tau=0.5,
                          lmbda=1e-2,
                          lr=1e-3,
                          seed=seed):
    
    optimizer = nnx.Optimizer(mask_model, optax.adam(lr), wrt=nnx.Param)

    key, subkey = jax.random.split(key)
    import numpy as np
    rng = np.random.default_rng(seed)

    for step in range(num_train_steps):

        (observations, actions, rewards, next_observations, terminations)  = rb.sample_batch(batch_size, rng)
        
        # Q: is it better to mimic only the choosen action q value or q values for all actions in a state?

        # Build masked Q-network
        masked_q = apply_mask_to_q(q, mask_model, tau, subkey)

        masked_q_sa = q_sa_from_network(
            masked_q,
            observations,
            actions,
        )
        q_sa = q_sa_from_network(
            q,
            observations,
            actions,
        )
        
        mask_params = nnx.state(mask_model)
        # total_loss, (q_diff_loss, l0_val) = mask_loss_fn(mask_params, q_sa, masked_q_sa, lmbda)  # for debug
        (total_loss, (q_diff_loss, l0_val)), grads = mask_loss_grad_fn(
            mask_params, q_sa, masked_q_sa, lmbda
        )
        # print(f"mask logits:\n{mask_params}")
        # print(f"grads:\n{grads}")

        optimizer.update(mask_model, grads)

        # train_step = partial(train_step_with_loss, ddqn_loss)
        # train_step = partial(nnx.jit, static_argnames=("gamma",))(train_step)        
        # def train_step_with_loss(
        #     loss, optimizer: nnx.Optimizer, q: nnx.Module, *args, **kwargs
        # ) -> tuple[float, float]:
        #     grad_fn = nnx.value_and_grad(loss, argnums=0, has_aux=True)
        #     value, grad = grad_fn(q, *args, **kwargs)
        #     optimizer.update(q, grad)
        # return value

        if step % 500 == 0:
            print(f"step {step}: loss={total_loss:.4f}, q_diff_loss={q_diff_loss:.4f}, l0={l0_val:.4f}")
            print(f"mask logits:\n{mask_params}")

    return mask_model

#----------- Plot -----------
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_binarized_state(state):
    layer_names = list(state.keys())
    n_layers = len(layer_names)

    # Choose subplot grid automatically
    ncols = math.ceil(math.sqrt(n_layers))
    nrows = math.ceil(n_layers / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = np.array(axes).reshape(-1)  # flatten in case of 2D grid

    for i, layer_name in enumerate(layer_names):
        arr = np.array(state[layer_name]['value'])  # JAX → NumPy

        axes[i].imshow(arr, cmap="gray", aspect="auto")
        axes[i].set_title(layer_name)
        axes[i].set_xlabel("Output units")
        axes[i].set_ylabel("Input units")

    # Hide unused subplots (if grid > layers)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# ---------- Usage ----------
train_mask = True
save_final_mask = True
load_final_mask = not train_mask

def binarize_state(state):
    return state.map(
        lambda path, var: var.replace(
            value=(jax.nn.sigmoid(var.value) > 0.5).astype(jnp.float32)
        )
    )

if train_mask:
    optimized_mask_model = optimize_mask_with_rb(q, mask_model, rb, key)
    print(f"optimized_mask_model:\n{optimized_mask_model}")

    # Final hard mask and pruned Q-net
    mask_params = nnx.state(optimized_mask_model)
    final_masks_params = binarize_state(mask_params)
    state_final_masks = nnx.state(final_masks_params)
    print(f"state_final_masks:\n{state_final_masks}")
    plot_binarized_state(state_final_masks)

mask_file = f"task_{subtask}_masks.pkl"
logits_file = f"task_{subtask}_logits.pkl"
mask_path = os.path.expanduser(
    f"~/workspace/XRL/ocean_subnet/{benchmark}_{grid_size}/uts/seed_{model_seed}/{ckpt_name}/{mask_file}"
)
logits_path = os.path.expanduser(
    f"~/workspace/XRL/ocean_subnet/{benchmark}_{grid_size}/uts/seed_{model_seed}/{ckpt_name}/{logits_file}"
)
os.makedirs(os.path.dirname(mask_path), exist_ok=True)

if load_final_mask:
    with open(mask_path, "rb") as f:
        state_final_masks = pickle.load(f)
        print(f"loaded_masks:\n{state_final_masks}")
    # Recreate model first
    final_masks_params = MaskNet(shapes, init_value=0.9)
    nnx.update(final_masks_params, state_final_masks)

    plot_binarized_state(state_final_masks)

if save_final_mask:
    with open(mask_path, "wb") as f:
        pickle.dump(state_final_masks, f)
    with open(logits_path, "wb") as f:
        pickle.dump(mask_params, f)

# TODO: apply final mask directly without drawing hard concrete sample
key, subkey = jax.random.split(key)
pruned_q_net = apply_mask_to_q(q, final_masks_params, tau=1e-6, key=subkey)

subnet_policy = get_policy_from_q_net(pruned_q_net)
print(f"evaluate subnet for task {subtask}")
evaluate_policy_on_task(subnet_policy, task=subtask, render_mode="None")

