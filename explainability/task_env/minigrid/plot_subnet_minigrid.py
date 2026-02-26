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

env = make_ocean_env(color = "red", render_mode = "human")

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
def evaluate_policy_on_task(policy, task="all"):
    # COLOR_NAMES = ["blue"]
    if task == "all":
        task = COLOR_NAMES
    else:
        task = task if isinstance(task, list) else [task]
    
    for color in task:
        print(f"color:{color}")
        assert color in COLOR_NAMES

        eval_env = make_ocean_env(color = color, render_mode = "human")
        _ = eval_policy(color, eval_env, policy, verbose=True, num_episode=1)

# evaluate_policy_on_task(policy)

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


def init_mask_logits(shapes, init_value=0.9):
    mask_logits = {}
    for key, shape in shapes.items():
        mask_logits[key] = jnp.full(shape, init_value)
    return mask_logits


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


def apply_mask_to_layer(layer_key, params, mask_logits, tau, key):
    """Apply hard‑concrete mask to all params in a layer."""
    new_params = {}

    for pname, pvalue in params.items():
        if pname == "kernel":
            key, subkey = jax.random.split(key)
            logits = mask_logits[layer_key]
            mask = hard_concrete_sample(logits, tau, subkey)
            # print(f"mask:\n{mask}")
            new_params[pname] = pvalue.value * mask
        else:
            new_params[pname] = pvalue

    return new_params, key

def apply_mask_to_q(q, mask_logits, tau, key):
    state = nnx.state(q)
    new_state = {}

    for layer_name, params in state.items():

        if layer_name == "hidden_layers":
            new_state[layer_name] = {}

            for idx, layer_params in params.items():
                layer_key = f"hidden_layer_{idx}"
                new_state[layer_name][idx], key = apply_mask_to_layer(
                    layer_key, layer_params, mask_logits, tau, key
                )

        elif layer_name == "output_layer":
            layer_key = "output_layer"
            new_state[layer_name], key = apply_mask_to_layer(
                layer_key, params, mask_logits, tau, key
            )

        else:
            raise ValueError(f"Unexpected layer: {layer_name}")

    masked_q_net = get_mlp_with_state(env, nnx.state(new_state))
    return masked_q_net


# 1. Get shapes
shapes = get_kernel_shapes(q)

# 2. Create trainable mask logits
mask_logits = init_mask_logits(shapes)
print(f"mask_logits:\n{mask_logits}")

# 3. Use mask during forward pass
key, subkey = jax.random.split(key)
masked_q_net = apply_mask_to_q(q, mask_logits, tau=0.5, key=subkey)

# # 4. Masked q net
# # hard masked in forward pass, soft masked in back prop
# masked_policy = get_policy_from_q_net(masked_q_net)
# evaluate_policy_on_task(masked_policy)


# ---------- L0 regularization over mask logits ----------

def l0_regularization(mask_logits, lmbda):
    reg = 0.0
    for logits in mask_logits.values():
        reg += jnp.sum(jax.nn.sigmoid(logits))
    return lmbda * reg

# ---------- Simple rollout + replay buffer utilities ----------

import warnings
from tqdm import tqdm
from rl_blox.blox.replay_buffer import ReplayBuffer

task_list = ["blue"] # "red", "green", "blue", "purple", "yellow", "grey"
# task_list = ["red", "green", "blue", "yellow"] # "red", "green", "blue", "purple", "yellow", "grey"


# ---------- Usage ----------
import pickle

for subtask in task_list:
    subtask_file = f"task_{subtask}_masks.pkl"

    with open(subtask_file, "rb") as f:
        final_masks = pickle.load(f)
        print(f"loaded_masks:\n{final_masks}")
    
    # TODO: save each mask separately


key, subkey = jax.random.split(key)
pruned_q_net = apply_mask_to_q(q, final_masks, tau=1e-6, key=subkey)

# subnet_policy = get_policy_from_q_net(pruned_q_net)
# evaluate_policy_on_task(subnet_policy, task=subtask)


# ---------- Plot ----------

import numpy as np
import matplotlib.pyplot as plt

def print_mask_sparsity(final_masks):
    """
    Print sparsity (percentage of zeros) for each mask.
    """
    for name, mask in final_masks.items():
        m = np.array(mask)
        sparsity = 1.0 - m.mean()
        print(f"{name}: shape={m.shape}, sparsity={sparsity:.3f}")

def visualize_2d_mask(name, mask):
    """
    Visualize a 2D mask as a heatmap.
    """
    mask_np = np.array(mask)

    plt.figure(figsize=(6, 4))
    plt.imshow(mask_np, cmap="gray_r", aspect="auto")
    plt.colorbar(label="Mask value")
    plt.title(f"Mask for layer: {name}")
    plt.xlabel("Output units")
    plt.ylabel("Input units")
    plt.show()

def visualize_1d_mask(name, mask):
    """
    Visualize a 1D mask as a stem plot.
    """
    mask_np = np.array(mask).flatten()

    plt.figure(figsize=(10, 2))
    plt.stem(mask_np, use_line_collection=True)
    plt.title(f"1D mask for layer: {name}")
    plt.ylim(-0.1, 1.1)
    plt.show()

def visualize_all_masks(final_masks):
    """
    Automatically visualize all masks in the dictionary.
    Chooses 1D or 2D visualization based on mask shape.
    """
    for name, mask in final_masks.items():
        m = np.array(mask)

        if m.ndim == 1:
            visualize_1d_mask(name, m)
        elif m.ndim == 2:
            visualize_2d_mask(name, m)
        else:
            print(f"Skipping {name}: unsupported shape {m.shape}")

# print_mask_sparsity(final_masks)
# visualize_all_masks(final_masks)

import numpy as np
import matplotlib.pyplot as plt
import math

def plot_masks_grid(final_masks):
    """
    Plot all masks in a grid layout.
    Each mask gets its own subplot.
    Works for 1D and 2D masks (MLP layers).
    """
    layer_names = list(final_masks.keys())
    num_layers = len(layer_names)

    # Grid size: square-ish layout
    cols = math.ceil(math.sqrt(num_layers))
    rows = math.ceil(num_layers / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")  # hide empty axes by default

    for idx, name in enumerate(layer_names):
        mask = np.array(final_masks[name])
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        if mask.ndim == 1:
            # 1D mask → stem plot
            ax.stem(mask, use_line_collection=True)
            ax.set_ylim(-0.1, 1.1)
            ax.set_title(name)
        elif mask.ndim == 2:
            # 2D mask → heatmap
            ax.imshow(mask, cmap="gray_r", aspect="auto")
            ax.set_title(name)
        else:
            ax.text(0.5, 0.5, f"Unsupported shape {mask.shape}",
                    ha="center", va="center")
            ax.set_title(name)

        ax.axis("on")

    plt.tight_layout()
    plt.show()

# plot_masks_grid(final_masks)


import numpy as np
import matplotlib.pyplot as plt
import math

def plot_masks_subfigures(final_masks):
    """
    Plot each mask in its own subplot, all inside one figure.
    Works for 1D and 2D masks (MLP layers).
    """
    layer_names = list(final_masks.keys())
    num_layers = len(layer_names)

    # Choose a grid layout that is as square as possible
    cols = math.ceil(math.sqrt(num_layers))
    rows = math.ceil(num_layers / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    # Turn off all axes initially
    for ax in axes.flat:
        ax.axis("off")

    # Fill in each subplot
    for idx, name in enumerate(layer_names):
        mask = np.array(final_masks[name])
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        if mask.ndim == 1:
            # 1D mask → stem plot
            ax.stem(mask, use_line_collection=True)
            ax.set_ylim(-0.1, 1.1)
            ax.set_title(name)
            ax.set_xlabel("Unit index")
            ax.set_ylabel("Mask value")
        elif mask.ndim == 2:
            # 2D mask → heatmap
            ax.imshow(mask, cmap="gray_r", aspect="auto")
            ax.set_title(name)
            ax.set_xlabel("Output units")
            ax.set_ylabel("Input units")
        else:
            ax.text(0.5, 0.5, f"Unsupported shape {mask.shape}",
                    ha="center", va="center")
            ax.set_title(name)

        ax.axis("on")

    plt.tight_layout()
    plt.show()

# plot_masks_subfigures(final_masks)

import numpy as np
import matplotlib.pyplot as plt

def plot_256x256_mask(final_masks):
    """
    Find and plot the mask whose shape is (256, 256).
    """
    found = False

    for name, mask in final_masks.items():
        mask_np = np.array(mask)

        if mask_np.shape == (256, 256):
            found = True
            plt.figure(figsize=(6, 6))
            plt.imshow(mask_np, cmap="gray_r", aspect="equal")
            plt.colorbar(label="Mask value")
            plt.title(f"Mask for layer: {name} (256×256)")
            plt.xlabel("Output units")
            plt.ylabel("Input units")
            plt.show()

    if not found:
        print("No mask with shape (256, 256) found.")

# plot_256x256_mask(final_masks)


import numpy as np
import matplotlib.pyplot as plt
import math

def plot_all_masks(final_masks):
    """
    Plot each mask in its own subplot inside a single figure.
    """
    layer_names = list(final_masks.keys())
    num_layers = len(layer_names)

    cols = math.ceil(math.sqrt(num_layers))
    rows = math.ceil(num_layers / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for idx, name in enumerate(layer_names):
        mask = np.array(final_masks[name])
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        if mask.ndim == 2:
            ax.imshow(mask, cmap="gray_r", aspect="auto")
            ax.set_title(name)
            ax.set_xlabel("Output units")
            ax.set_ylabel("Input units")
        elif mask.ndim == 1:
            ax.stem(mask, use_line_collection=True)
            ax.set_ylim(-0.1, 1.1)
            ax.set_title(name)
        else:
            ax.text(0.5, 0.5, f"Unsupported shape {mask.shape}",
                    ha="center", va="center")

        ax.axis("on")

    plt.tight_layout()
    plt.show()

# plot_all_masks(final_masks)

# print(jax.tree_util.tree_structure(final_masks))
