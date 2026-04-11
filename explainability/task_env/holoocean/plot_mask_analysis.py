import os
# from find_many_objects_env import make_ocean_env
# from minigrid.core.constants import COLOR_TO_IDX

import flax.nnx as nnx
import numpy as np

seed = 49  # random seed for np and jax
plot_mask_only = True

# ---------------------------
# (1) HoloOcean env params
# ---------------------------
# TODO: use real environment

subtasks = ["red", "blue", "green", "black"]
COLOR_NAMES = subtasks

# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "blue": 1, "green": 2, "black": 3}

num_current_context = 2
num_task_context = 4


# ---------------------------
# (1) Load trained network
# ---------------------------
import orbax.checkpoint as ocp
from rl_blox.blox.function_approximator.mlp import MLP

# Choose ckpt from pre-trained full q-net
experiment = "holoocean"
randomize_env = True
seed_num = 0
file_name = "_minigrid_holoocean_medium_DDQN-UTS_1774215135.9690797_q_step_000250000_epoch_250000"
model_hidden_nodes = [128, 128]


# Set up ckpt path
ckpt_full_net = os.path.expanduser(
    f"~/workspace/XRL/ocean_trained_model/{experiment}/UTS/seed_{seed_num}/{file_name}"
)
step = int(file_name.split("_")[-3])
plot_info = f"{experiment}, seed {seed_num}, step {step}"

# Set up environment to get input/output shapes
subtask = subtasks[0]
# env = make_ocean_env(subtask)

# Recreate MLP with same architecture as training
hparams_model = dict(
    n_features=50,
    n_outputs=5,
    activation="relu",
    hidden_nodes=model_hidden_nodes,
)

# Restore q net from checkpoint
abstract_mlp = MLP(rngs=nnx.Rngs(seed), **hparams_model)
graphdef, abstract_state = nnx.split(abstract_mlp)


checkpointer = ocp.StandardCheckpointer()

#--------------
weights_dict = {}

for subtask in subtasks:
    # env = make_ocean_env(subtask)
    ckpt_subnet = os.path.expanduser(
        f"~/workspace/XRL/ocean_subnet/{experiment}/UTS/seed_{seed_num}/step_{step}/subnetwork_{subtask}"
    )
    restored_model = checkpointer.restore(ckpt_subnet, abstract_state)
    q_net_sub = nnx.merge(graphdef, restored_model)
    print("Loaded subnetwork {subtask} checkpoint.")

    # Collect weight matrices (hidden + output)
    weights = []
    for layer in q_net_sub.hidden_layers:
        weights.append(np.array(layer.kernel.value))
    weights.append(np.array(q_net_sub.output_layer.kernel.value))

    weights_dict[subtask] = weights

print("Weights collected for all subtasks.")


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------
# Inputs
# ---------------------------
subtask_list = list(weights_dict.keys())
num_layers = len(weights_dict[subtask_list[0]])

# ---------------------------
# Colors
# ---------------------------
subtask_colors = {
    "red":   [1, 0, 0],
    "blue":  [0, 0, 1],
    "green": [0, 0.7, 0],
    "black": [0, 0, 0],
}

LIGHT_GREY = [0.93, 0.93, 0.93]
YELLOW = [1.0, 0.85, 0.2]

shared_weights_color = LIGHT_GREY
partially_shared_weights_color = YELLOW

# ---------------------------
# Build visualization
# ---------------------------
layers_vis = []
titles = []

for i in range(num_layers):

    # stack: (num_subtasks, H, W)
    stacked = np.stack(
        [weights_dict[sub][i] for sub in subtask_list],
        axis=0
    )

    used_any = (stacked != 0).any(axis=0)
    used_all = (stacked != 0).all(axis=0)

    h, w = stacked.shape[1], stacked.shape[2]

    # base image
    rgb = np.ones((h, w, 3))  # unused = white

    # ---------------------------
    # shared regions
    # ---------------------------
    rgb[used_any & (~used_all)] = partially_shared_weights_color
    rgb[used_all] = shared_weights_color

    # ---------------------------
    # exclusive regions
    # ---------------------------
    for s_idx, subtask in enumerate(subtask_list):

        current = stacked[s_idx]
        used_current = (current != 0)

        others = np.delete(stacked, s_idx, axis=0)
        used_others = others.any(axis=0)

        mask_exclusive = used_current & (~used_others)

        rgb[mask_exclusive] = subtask_colors[subtask]

    layers_vis.append(rgb)

    layer_name = "output layer" if i == num_layers - 1 else f"hidden layer {i}"
    titles.append(layer_name)

# ---------------------------
# Plot
# ---------------------------
font_size = 24
scale = 0.1
w_space = 0.25

width_ratios = [W.shape[1] for W in layers_vis]
fig_width = sum(width_ratios) * scale
fig_height = max(W.shape[0] for W in layers_vis) * scale

fig = plt.figure(figsize=(fig_width, fig_height))
gs = fig.add_gridspec(1, len(layers_vis), width_ratios=width_ratios, wspace=w_space)

axes_list = []

for i, W in enumerate(layers_vis):

    ax = fig.add_subplot(gs[0, i])
    ax.imshow(W, aspect='equal')
    axes_list.append(ax)

    if i == 0:
        ax.set_xlabel("output neuron index", fontsize=font_size)
        ax.set_ylabel("input neuron index", fontsize=font_size)

    h, w = W.shape[:2]
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    ax.set_title(titles[i], fontsize=font_size)

# ---------------------------
# Legend
# ---------------------------
legend_patches = [
    mpatches.Patch(color=shared_weights_color, label='used by all subtasks'),
    mpatches.Patch(color=partially_shared_weights_color, label='shared (partial)'),
]

for subtask in subtask_list:
    legend_patches.append(
        mpatches.Patch(color=subtask_colors[subtask], label=f'exclusive ({subtask})')
    )

legend_patches.append(
    mpatches.Patch(facecolor='white', edgecolor='grey', label='unused')
)

fig.legend(
    handles=legend_patches,
    loc='lower center',
    ncol=2,
    fontsize=font_size
)

fig.suptitle("Weights Analysis Across All Subtasks", fontsize=font_size)

# ---------------------------
# Save
# ---------------------------
save_fig = True
if save_fig:
    path = os.path.expanduser('~/Pictures/all_subtasks_shared_exclusive.pdf')
    plt.savefig(path)

# plt.show()