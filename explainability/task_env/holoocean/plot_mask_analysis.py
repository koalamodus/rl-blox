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

    layer_name = "output layer" if i == num_layers - 1 else f"hidden layer {i}"
    titles.append(layer_name)

# ---------------------------
# Plot params
# ---------------------------
save_fig = True

font_size = 24
w_space = 0.25
label_pad = 10
title_pad = 15
scale = 0.1
margin_width_ratio = 0.0
annotation_line_width = 2.0
legend_pos = [0.2, 0.7]

# ---------------------------
# Plot
# ---------------------------

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

cmap = ListedColormap(["white", "green", "#C2E1BCB7"])

width_ratios = [W.shape[1] for W in layers_vis]
fig_width = (sum(width_ratios) + margin_width_ratio) * scale
fig_height = max(W.shape[0] for W in layers_vis) * scale

fig = plt.figure(figsize=(fig_width, fig_height))
gs = fig.add_gridspec(1, len(layers_vis), width_ratios=width_ratios, wspace=w_space)

axes_list = []

for i, W in enumerate(layers_vis):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(W, cmap=cmap, aspect='equal', vmin=0, vmax=2)
    axes_list.append(ax)

    if i == 0:
        ax.set_xlabel("output neuron index", fontsize=font_size, labelpad=label_pad)
        ax.set_ylabel("input neuron index", fontsize=font_size, labelpad=label_pad)

    h, w = W.shape
    ax.set_xlim(-0.5, w-0.5)
    ax.set_ylim(h-0.5, -0.5)

    if i == len(layers_vis)-1:
        ax.set_xticks([0, w-1])
        ax.set_xticklabels([0, w-1])

    ax.set_title(titles[i], fontsize=font_size, pad=title_pad)
    ax.tick_params(axis='both', labelsize=font_size)

    # --- Add annotation for Hidden Layer 0 ---
    if i == 0:
        x_axes = 1.02
        y_axes = 1 - (num_current_context + num_task_context/2) / h

        overlap = 0.003

        turning_point_0 = [x_axes + 0.03, y_axes]
        turning_point_1 = [x + y for x, y in zip(turning_point_0, [0, 0.5])]
        turning_point_2 = [x + y for x, y in zip(turning_point_1, [-0.15, 0])]
        text_pos = [x + y for x, y in zip(turning_point_2, [-0.6, 0])]

        ax.annotate(
            "",
            xy=(x_axes, y_axes),
            xycoords=ax.transAxes,
            xytext=(turning_point_0[0], turning_point_0[1]),
            textcoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="-[,widthB=0.4,lengthB=0.3",
                color="black",
                linewidth=annotation_line_width,
            ),
            fontsize=font_size,
            color='black',
            va='center',
            clip_on=False
        )

        ax.annotate(
            "",
            xy=(turning_point_0[0] - overlap, turning_point_0[1] - overlap),                  # arrow head (end)
            xycoords=ax.transAxes,
            xytext=(turning_point_1[0] - overlap, turning_point_1[1] + overlap),        # arrow tail (start) 0.1 above
            textcoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="-",  # bracket-style line
                color="black",
                linewidth=annotation_line_width,
            ),
            fontsize=font_size,
            color='black',
            va='center',
            clip_on=False
        )

        ax.annotate(
            "",
            xy=(turning_point_1[0], turning_point_1[1]),                  # arrow head (end)
            xycoords=ax.transAxes,
            xytext=(turning_point_2[0], turning_point_2[1]),        # arrow tail (start) 0.1 above
            textcoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="<-",  # bracket-style line
                color="black",
                linewidth=annotation_line_width,
            ),
            fontsize=font_size,
            color='black',
            va='center',
            clip_on=False
        )

        ax.annotate(
            "context encoding of subtasks",
            xy=(text_pos[0], text_pos[1]),                  # arrow head (end)
            xycoords=ax.transAxes,
            xytext=(text_pos[0], text_pos[1]),        # arrow tail (start) 0.1 above
            textcoords=ax.transAxes,
            fontsize=font_size,
            color='black',
            va='center',
            clip_on=False
        )


    # --- Add annotation for Output layer ---
    if i == len(layers_vis) - 1:

        # middle of axis (normalized coords)
        x_middle = 0.5
        y_bracket = -0.07
        y_text = -0.15

        ax.annotate(
            "actions",
            xy=(x_middle, y_bracket),
            xycoords=ax.transAxes,
            xytext=(x_middle, y_text),
            textcoords=ax.transAxes,
            ha="center",
            va="top",
            arrowprops=dict(
                arrowstyle="<-[,widthB=0.5,lengthB=0.3",
                color="black",
                linewidth=annotation_line_width,
            ),
            fontsize=font_size,
            color='black',
            clip_on=False
        )

# Align bottoms
bottom_y = 0.2
for ax in axes_list:
    pos = ax.get_position()
    height = pos.height  # or pos.y1 - pos.y0
    ax.set_position([pos.x0, bottom_y, pos.width, height])

# Legend
legend_patches = [
    mpatches.Patch(color="#C2E1BCB7", label='shared weights'),
    mpatches.Patch(color='green', label='task-specific weights'),
    mpatches.Patch(facecolor='white', edgecolor='grey', linewidth=0.8, label='masked (unused) weights')
]

fig.legend(handles=legend_patches,
           loc='lower center',
           bbox_to_anchor=(legend_pos[0], legend_pos[1]),
           ncol=1,
           fontsize=font_size,
           handlelength=1.0,                # length of the colored box
           handleheight=1.0,
           )

fig.suptitle("shared weights across subtasks", fontsize=font_size)
fig.text(
        0.5,
        0.01,
        f"{plot_info}",
        ha='center',
        va='bottom',
        fontsize=font_size
    )

# plt.subplots_adjust(top=0.9, bottom=0.1, right=0.9, left=0.1)
# plt.tight_layout()

if save_fig:
    file_name = "holoocean_shared_weights.pdf"
    path = os.path.expanduser(f'~/Pictures/{file_name}')
    plt.savefig(path)

# plt.show()

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

        layer_name = "output layer" if i == num_layers - 1 else f"hidden layer {i}"
        titles.append(layer_name)

    # ---------------------------
    # Plot
    # ---------------------------

    width_ratios = [W.shape[1] for W in layers_vis]
    fig_width = (sum(width_ratios) + margin_width_ratio) * scale
    fig_height = max(W.shape[0] for W in layers_vis) * scale

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(1, len(layers_vis), width_ratios=width_ratios, wspace=w_space)
    axes_list = []

    for i, W in enumerate(layers_vis):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(W, cmap=cmap, aspect='equal', vmin=0, vmax=3)
        axes_list.append(ax)

        if i == 0:
            ax.set_xlabel("output neuron index", fontsize=font_size, labelpad=label_pad)
            ax.set_ylabel("input neuron index", fontsize=font_size, labelpad=label_pad)

        h, w = W.shape
        ax.set_xlim(-0.5, w-0.5)
        ax.set_ylim(h-0.5, -0.5)

        if i == len(layers_vis)-1:
            ax.set_xticks([0, w-1])
            ax.set_xticklabels([0, w-1])

        ax.set_title(titles[i], fontsize=font_size, pad=title_pad)
        ax.tick_params(axis='both', labelsize=font_size)
    
        # --- Add annotation for Hidden Layer 0 ---
        if i == 0:
            context = COLOR_TO_IDX[subtask]

            x_axes = 1.0
            y_axes = 1 - (num_current_context + context + 0.5) / h

            overlap = 0.003

            turning_point_0 = [x_axes + 0.03, y_axes]
            turning_point_1 = [x + y for x, y in zip(turning_point_0, [0, 0.5])]
            turning_point_2 = [x + y for x, y in zip(turning_point_1, [-0.1, 0])]
            text_pos = [x + y for x, y in zip(turning_point_2, [-0.7, 0])]
            
            ax.annotate(
                "",
                xy=(x_axes, y_axes),
                xycoords=ax.transAxes,
                xytext=(turning_point_0[0], turning_point_0[1]),
                textcoords=ax.transAxes,
                arrowprops=dict(
                    arrowstyle="-",
                    color='black',
                    linewidth=annotation_line_width,
                ),
                fontsize=font_size,
                color='black',
                va='center',
                clip_on=False                       # allow drawing outside axis
            )

            ax.annotate(
                "",
                xy=(turning_point_0[0] - overlap, turning_point_0[1] - overlap),                  # arrow head (end)
                xycoords=ax.transAxes,
                xytext=(turning_point_1[0] - overlap, turning_point_1[1] + overlap),        # arrow tail (start) 0.1 above
                textcoords=ax.transAxes,
                arrowprops=dict(
                    arrowstyle="-",  # bracket-style line
                    color="black",
                    linewidth=annotation_line_width,
                ),
                fontsize=font_size,
                color='black',
                va='center',
                clip_on=False
            )

            ax.annotate(
                "",
                xy=(turning_point_1[0], turning_point_1[1]),                  # arrow head (end)
                xycoords=ax.transAxes,
                xytext=(turning_point_2[0], turning_point_2[1]),        # arrow tail (start) 0.1 above
                textcoords=ax.transAxes,
                arrowprops=dict(
                    arrowstyle="<-",  # bracket-style line
                    color="black",
                    linewidth=annotation_line_width,
                ),
                fontsize=font_size,
                color='black',
                va='center',
                clip_on=False
            )

            ax.annotate(
                f"context encoding of subtask {subtask}",
                xy=(text_pos[0], text_pos[1]),                  # arrow head (end)
                xycoords=ax.transAxes,
                xytext=(text_pos[0], text_pos[1]),        # arrow tail (start) 0.1 above
                textcoords=ax.transAxes,
                fontsize=font_size,
                color='black',
                va='center',
                clip_on=False
            )
    
    # Align bottoms
    bottom_y = 0.2
    for ax in axes_list:
        pos = ax.get_position()
        height = pos.height  # or pos.y1 - pos.y0
        ax.set_position([pos.x0, bottom_y, pos.width, height])

    # Legend
    legend_patches = [
        mpatches.Patch(color="#C2E1BCB7", label='shared weights across tasks'),
        mpatches.Patch(color='green', label='task-specific weights'),
        # mpatches.Patch(color='grey', label='used in other subtasks'),
        mpatches.Patch(facecolor='white', edgecolor='grey', linewidth=0.8, label='masked (unused) weights')
    ]

    fig.legend(handles=legend_patches,
               loc='lower center',
               bbox_to_anchor=(legend_pos[0], legend_pos[1]),
               ncol=1,
               fontsize=font_size,
               handlelength=1.0,                # length of the colored box
               handleheight=1.0,
               )

    fig.suptitle(f"subnetwork {subtask}", fontsize=font_size)
    fig.text(
        0.5,
        0.01,
        f"{plot_info}",
        ha='center',
        va='bottom',
        fontsize=font_size
    )

    if save_fig:
        file_name = f"holoocean_subnetwork_{subtask}.pdf"
        path = os.path.expanduser(f'~/Pictures/{file_name}')
        plt.savefig(path)

    # plt.show()