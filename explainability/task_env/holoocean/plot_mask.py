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

# ---------------------------
# (1) Load trained network
# ---------------------------
import orbax.checkpoint as ocp
from rl_blox.blox.function_approximator.mlp import MLP

# Choose ckpt from pre-trained full q-net

# experiment = "reefshield_random_medium"
# randomize_env = True
# seed_num = 0
# file_name = "_minigrid_reefshield_random_medium_DDQN-UTS_1773787402.9864454_q_step_005000000_epoch_5000000"
# model_hidden_nodes = [32, 32]

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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------
# Shared weights across subtasks
# ---------------------------

subtask_list = list(weights_dict.keys())
num_layers = len(weights_dict[subtask_list[0]])

layers_vis = []
titles = []

for i in range(num_layers):
    stacked = np.stack([weights_dict[sub][i] for sub in subtask_list], axis=0)

    # Masks
    used_any = (stacked != 0).any(axis=0)     # union
    used_all = (stacked != 0).all(axis=0)     # intersection

    vis = np.zeros_like(used_any, dtype=int)
    vis[used_any & (~used_all)] = 1
    vis[used_all] = 2

    layers_vis.append(vis)

    layer_name = "Output Layer" if i == num_layers - 1 else f"Hidden Layer {i}"
    titles.append(layer_name)

# ---------------------------
# Plot
# ---------------------------

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

cmap = ListedColormap(["white", "green", "#C2E1BCB7"])

width_ratios = [W.shape[1] for W in layers_vis]
scale = 0.1
fig_width = sum(width_ratios) * scale
fig_height = max(W.shape[0] for W in layers_vis) * scale

fig = plt.figure(figsize=(fig_width, fig_height))
gs = fig.add_gridspec(1, len(layers_vis), width_ratios=width_ratios, wspace=0.3)

for i, W in enumerate(layers_vis):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(W, cmap=cmap, aspect='equal', vmin=0, vmax=2)

    if i == 0:
        ax.set_xlabel("Output Neuron Index", fontsize=10)
        ax.set_ylabel("Input Neuron Index", fontsize=10)

    h, w = W.shape
    ax.set_xlim(-0.5, w-0.5)
    ax.set_ylim(h-0.5, -0.5)

    if i == len(layers_vis)-1:
        ax.set_xticks([0, w-1])
        ax.set_xticklabels([0, w-1])

    ax.set_title(titles[i], fontsize=10)
    ax.tick_params(axis='both', labelsize=8)

    # --- Add annotation for Hidden Layer 0 ---
    if i == 0:
        x_axes = 1.02
        y_axes = (6/2) / h

        ax.annotate(
            "context encoding of subtasks",
            xy=(x_axes, y_axes),
            xycoords=ax.transAxes,
            xytext=(x_axes + 0.15, y_axes),
            textcoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="<-[,widthB=1.5,lengthB=0.3",
                color="black",
                linewidth=1.0,
            ),
            fontsize=9,
            color='black',
            va='center',
            clip_on=False
        )
    # --- Add annotation for Output layer ---
    if i == len(layers_vis) - 1:

        # middle of axis (normalized coords)
        x_middle = 0.5
        y_bracket = -0.12
        y_text = -0.25

        ax.annotate(
            "actions",
            xy=(x_middle, y_bracket),
            xycoords=ax.transAxes,
            xytext=(x_middle, y_text),
            textcoords=ax.transAxes,
            ha="center",
            va="top",
            arrowprops=dict(
                arrowstyle="<-[,widthB=0.75,lengthB=0.3",
                color="black",
                linewidth=1.0,
            ),
            fontsize=9,
            color='black',
            clip_on=False
        )

# Legend
legend_patches = [
    mpatches.Patch(color="#C2E1BCB7", label='Shared weights'),
    mpatches.Patch(color='green', label='Task-specific weights'),
    mpatches.Patch(facecolor='white', edgecolor='grey', linewidth=0.8, label='Masked (unused) weights')
]

fig.legend(handles=legend_patches,
           loc='lower center', bbox_to_anchor=(0.75, 0.8),
           ncol=1, fontsize=8)

fig.suptitle("Shared weight across subtasks", fontsize=14)
fig.text(
        0.5,
        0.01,
        f"{plot_info}",
        ha='center',
        va='bottom',
        fontsize=8
    )
    # fig.text(
    #     0.5,
    #     0.2,
    #     mask_info,
    #     ha='left',
    #     va='bottom',
    #     fontsize=8
    # )
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# ---------------------------
# Subnetwork weights

# green → weights used by this subtask but not used in all subtasks
# white → weights used by other subtasks (but not this one)
# light green → weights used by all subtasks
# white → unused everywhere
# ---------------------------

subtask_list = list(weights_dict.keys())
num_layers = len(weights_dict[subtask_list[0]])

# 0=black, 1=green, 2=pink, 3=light green
cmap = ListedColormap(["white", "green", "white", "#C2E1BCB7"])

for s_idx, subtask in enumerate(subtask_list):

    layers_vis = []
    titles = []

    for i in range(num_layers):
        stacked = np.stack([weights_dict[sub][i] for sub in subtask_list], axis=0)

        current = stacked[s_idx]
        others = np.delete(stacked, s_idx, axis=0)

        # Masks
        used_current = (current != 0)
        used_others = (others != 0).any(axis=0)
        used_all = (stacked != 0).all(axis=0)

        # Initialize → black
        vis = np.zeros_like(current, dtype=int)

        # 1. Used in ALL subtasks → light green
        vis[used_all] = 3

        # 2. Used in THIS subtask but NOT all → green
        vis[used_current & (~used_all)] = 1

        # 3. Used in OTHER subtasks but NOT this one → pink
        vis[(~used_current) & used_others] = 2

        layers_vis.append(vis)

        layer_name = "Output Layer" if i == num_layers - 1 else f"Hidden Layer {i}"
        titles.append(layer_name)

    # ---------------------------
    # Plot
    # ---------------------------

    width_ratios = [W.shape[1] for W in layers_vis]
    scale = 0.1
    fig_width = sum(width_ratios) * scale
    fig_height = max(W.shape[0] for W in layers_vis) * scale

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(1, len(layers_vis), width_ratios=width_ratios, wspace=0.3)

    for i, W in enumerate(layers_vis):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(W, cmap=cmap, aspect='equal', vmin=0, vmax=3)

        if i == 0:
            ax.set_xlabel("Output Neuron Index", fontsize=10)
            ax.set_ylabel("Input Neuron Index", fontsize=10)

        h, w = W.shape
        ax.set_xlim(-0.5, w-0.5)
        ax.set_ylim(h-0.5, -0.5)

        if i == len(layers_vis)-1:
            ax.set_xticks([0, w-1])
            ax.set_xticklabels([0, w-1])

        ax.set_title(titles[i], fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
    
        # --- Add annotation for Hidden Layer 0 ---
        if i == 0:
            context = COLOR_TO_IDX[subtask]
            # coordinates for the arrow tip (lowest row, rightmost column)
            arrow_x = w - 1
            arrow_y = h - 6 + context

            ax.annotate(
                f"context encoding of subtask {subtask}",
                xy=(arrow_x, arrow_y),                  # point of interest
                xytext=(arrow_x + 5, arrow_y + 0.5),    # text location (to the right)
                arrowprops=dict(arrowstyle="<-", facecolor='black'),
                fontsize=9,
                color='black'
            )

    # Legend
    legend_patches = [
        mpatches.Patch(color="#C2E1BCB7", label='Shared weights across all tasks'),
        mpatches.Patch(color='green', label='Task-specific weights'),
        # mpatches.Patch(color='grey', label='Used in other subtasks'),
        mpatches.Patch(facecolor='white', edgecolor='grey', linewidth=0.8, label='Masked (unused) weights')
    ]

    fig.legend(handles=legend_patches,
               loc='lower center', bbox_to_anchor=(0.75, 0.8),
               ncol=1, fontsize=8)

    fig.suptitle(f"Subnetwork {subtask}", fontsize=14)
    fig.text(
        0.5,
        0.01,
        f"{plot_info}",
        ha='center',
        va='bottom',
        fontsize=8
    )
    # fig.text(
    #     0.5,
    #     0.2,
    #     mask_info,
    #     ha='left',
    #     va='bottom',
    #     fontsize=8
    # )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

# ---------------------------
# Prepare masked layers
# ---------------------------
subtask_list = list(weights_dict.keys())
num_layers = len(weights_dict[subtask_list[0]])

masked_layers = []
titles = []

for i in range(num_layers):
    stacked = np.stack([weights_dict[sub][i] for sub in subtask_list], axis=0)
    zero_all_mask = (stacked == 0).all(axis=0)

    if plot_mask_only:
        # Mask-only plot: black = zero in all subtasks
        masked_array = np.ma.masked_array(np.ones_like(zero_all_mask), mask=zero_all_mask)
    else:
        # Weight-plot: average weight across subtasks, mask zeros in all subtasks
        avg_weights = stacked.mean(axis=0)
        masked_array = np.ma.masked_array(avg_weights, mask=zero_all_mask)

    masked_layers.append(masked_array)
    layer_name = "Output Layer" if i == num_layers - 1 else f"Hidden Layer {i}"
    titles.append(layer_name)

# ---------------------------
# Plot masked weights across all subtasks
# ---------------------------
if plot_mask_only:
    custom_cmap = LinearSegmentedColormap.from_list("mask_bw", ["white", "black"])
else:
    colors = [
        (0, "blue"),
        (0.49, "lightblue"),
        (0.5, "white"),
        (0.51, "lightcoral"),
        (1, "red")
    ]
    custom_cmap = LinearSegmentedColormap.from_list("highlight_zero", colors)

shared_mask_color = 'black'
subtask_mask_color = 'red' if plot_mask_only else 'yellow'

custom_cmap.set_bad(color=shared_mask_color)  # Black for masked (zero in all subtasks)

# Figure size and layout
width_ratios = [W.shape[1] for W in masked_layers] + [1.5]
scale = 0.1
fig_width = sum(width_ratios) * scale
fig_height = max(W.shape[0] for W in masked_layers) * scale

fig = plt.figure(figsize=(fig_width, fig_height))
gs = fig.add_gridspec(1, len(masked_layers)+1, width_ratios=width_ratios, wspace=0.5)

# Shared color scale for weight plots
if not plot_mask_only:
    abs_max = max(np.abs(W).max() for W in masked_layers)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
else:
    norm = None

axes = []
for i, W in enumerate(masked_layers):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(W, cmap=custom_cmap, norm=norm, aspect='equal')

    if i == 0:
        ax.set_xlabel("Output Neuron Index", fontsize=10)
        ax.set_ylabel("Input Neuron Index", fontsize=10)

    h, w = W.shape
    ax.set_xlim(-0.5, w-0.5)
    ax.set_ylim(h-0.5, -0.5)

    if i == len(masked_layers)-1:
        ax.set_xticks([0, w-1])
        ax.set_xticklabels([0, w-1])

    ax.set_title(titles[i], fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    axes.append(ax)

# Colorbar (only for weight-plot mode)
if not plot_mask_only:
    cax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Weight Value", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

# Figure title and bottom text
fig.suptitle(f"Masked weights across all subtasks", fontsize=14)
fig.text(
    0.5,
    0.01,
    f"{plot_info}",
    ha='center',
    va='bottom',
    fontsize=8
)

# Legend for masked weights
masked_patch = mpatches.Patch(color=shared_mask_color, label='Mask in all subtasks')
fig.legend(handles=[masked_patch], loc='lower center', bbox_to_anchor=(0.7, 0.10),
           ncol=1, fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---------------------------
# Plot masked weights for each subtasks
# ---------------------------

subtask_list = list(weights_dict.keys())
num_layers = len(weights_dict[subtask_list[0]])

for subtask in subtask_list:
    masked_layers = []
    titles = []

    # Compute total masked weights per layer
    layer_mask_info = []
    
    for i in range(num_layers):
        stacked = np.stack([weights_dict[sub][i] for sub in subtask_list], axis=0)
        zero_all_mask = (stacked == 0).all(axis=0)
        shared_mask = np.sum(zero_all_mask)

        current_weights = stacked[subtask_list.index(subtask)]
        black_mask = (current_weights == 0) & (~zero_all_mask)  # Black
        subtask_specific = np.sum(black_mask)
        layer_mask_info.append((shared_mask, subtask_specific))

        # Start with weights
        plot_array = np.copy(current_weights)
        
        # Create masked array: mask global zeros for special coloring
        masked_array = np.ma.masked_array(plot_array, mask=zero_all_mask)
        masked_layers.append((masked_array, black_mask, zero_all_mask))

        layer_name = "Output Layer" if i == num_layers - 1 else f"Hidden Layer {i}"
        titles.append(layer_name)
    
    mask_info = ""
    for i, (shared, subtask_only) in enumerate(layer_mask_info):
        layer_name = "Output Layer" if i == num_layers - 1 else f"Hidden Layer {i}"
        mask_info += f"{layer_name}: {shared} shared, {subtask_only} task-specific\n"


    # Plot
    width_ratios = [W.shape[1] for W, _, _ in masked_layers] + [1.5]
    scale = 0.1
    fig_width = sum(width_ratios) * scale
    fig_height = max(W.shape[0] for W, _, _ in masked_layers) * scale
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(1, len(masked_layers)+1, width_ratios=width_ratios, wspace=0.5)

    if plot_mask_only:
        custom_cmap = LinearSegmentedColormap.from_list("mask_bw", ["white", "black"])
        custom_cmap.set_bad(color=shared_mask_color)
    else:
        abs_max = max(np.abs(W).max() for W, _, _ in masked_layers)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        colors = [
            (0, "blue"),
            (0.49, "lightblue"),
            (0.5, "white"),
            (0.51, "lightcoral"),
            (1, "red")
        ]
        custom_cmap = LinearSegmentedColormap.from_list("highlight_zero", colors)
        custom_cmap.set_bad(color=shared_mask_color)

    axes = []
    for i, (W_masked, black_mask, grey_mask) in enumerate(masked_layers):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(W_masked, cmap=custom_cmap, norm=norm, aspect='equal')

        if plot_mask_only:
            # Overlay black for zeros in this subtask only
            black_overlay = np.zeros_like(W_masked, dtype=float)
            black_overlay[black_mask] = 1
            ax.imshow(black_overlay, cmap='Greys', alpha=1.0)

        subtask_overlay = np.ma.masked_where(~black_mask, black_mask)  # mask everything except current subtask zeros
        ax.imshow(subtask_overlay, cmap=LinearSegmentedColormap.from_list('subtask_mask', [subtask_mask_color, subtask_mask_color]), alpha=1.0)

        if i == 0:
            ax.set_xlabel("Output Neuron Index", fontsize=10)
            ax.set_ylabel("Input Neuron Index", fontsize=10)

        h, w = W_masked.shape
        ax.set_xlim(-0.5, w-0.5)
        ax.set_ylim(h-0.5, -0.5)

        if i == len(masked_layers)-1:
            ax.set_xticks([0, w-1])
            ax.set_xticklabels([0, w-1])

        ax.set_title(titles[i], fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        axes.append(ax)

    # Colorbar (only for weight-plot mode)
    if not plot_mask_only:
        cax = fig.add_subplot(gs[0, -1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Weight Value", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    # Legend
    patches = [
        mpatches.Patch(color=shared_mask_color, label='Mask across all subtasks'),
        mpatches.Patch(color=subtask_mask_color, label='Mask in this subtask')
    ]
    fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.7, 0.10),
               ncol=1, fontsize=8)

    fig.suptitle(f"Subtask {subtask} masked weights", fontsize=14)
    fig.text(
        0.5,
        0.01,
        f"{plot_info}",
        ha='center',
        va='bottom',
        fontsize=8
    )
    fig.text(
        0.5,
        0.2,
        mask_info,
        ha='left',
        va='bottom',
        fontsize=8
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()