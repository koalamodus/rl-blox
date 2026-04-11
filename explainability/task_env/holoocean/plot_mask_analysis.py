import os
# from find_many_objects_env import make_ocean_env
# from minigrid.core.constants import COLOR_TO_IDX

import flax.nnx as nnx
import numpy as np

seed = 49  # random seed for np and jax
plot_mask_only = True

# ---------------------------
# (0) HoloOcean env params
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
import seaborn as sns

# ---------------------------
# Inputs
# ---------------------------
subtask_list = list(weights_dict.keys())
num_layers = len(weights_dict[subtask_list[0]])

# ---------------------------
# Colors
# ---------------------------
palette = sns.color_palette("colorblind")

# Seaborn colorblind palette indices:
# 0: blue
# 1: orange
# 2: green
# 3: red (vermillion)
# 4: purple
# 5: brown
# 6: pink
# 7: grey
# 8: yellow-green (avoid as "yellow")

subtask_colors = {
    "red":   palette[3],  # vermillion
    "blue":  palette[0],  # blue
    "green": palette[2],  # green
    "black": palette[7],  # grey (closest usable)
}

LIGHT_GREY = [0.93, 0.93, 0.93]

shared_weights_color = LIGHT_GREY
partially_shared_weights_color = palette[6]  # pink


# ---------------------------
# Build visualization
# ---------------------------
layers_vis = []

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

# ---------------------------
# Plot
# ---------------------------
font_size = 26
scale = 0.1
w_space = 0.25
label_pad = 5
margin_width_ratio = 0.0
annotation_line_width = 2.0
legend_pos = [0.25, 0.55]

width_ratios = [W.shape[1] for W in layers_vis]
fig_width = (sum(width_ratios) + margin_width_ratio) * scale
fig_height = max(W.shape[0] for W in layers_vis) * scale

fig = plt.figure(figsize=(fig_width, fig_height))
gs = fig.add_gridspec(1, len(layers_vis), width_ratios=width_ratios, wspace=w_space)

axes_list = []
label_str = "neuron index"
xy_labels = [
    f"(state)\ninput layer {label_str}",
    f"first hidden layer {label_str}",
    f"second hidden layer {label_str}",
    f"output layer {label_str}\n(action)"
]
for i, W in enumerate(layers_vis):

    ax = fig.add_subplot(gs[0, i])
    ax.imshow(W, aspect='equal')
    axes_list.append(ax)

    ax.set_ylabel(
        xy_labels[i],
        fontsize=font_size,
        labelpad=label_pad
    )

    ax.set_xlabel(
        xy_labels[i+1],
        fontsize=font_size,
        labelpad=label_pad
    )

    h, w = W.shape[:2]
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    # ax.tick_params(axis='both', labelsize=font_size)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Add annotation for Hidden Layer 0 ---
    if i == 0:
        x_axes = 1.02
        y_axes = 1 - (num_current_context + num_task_context/2) / h

        overlap = 0.003

        turning_point_0 = [x_axes + 0.03, y_axes]
        turning_point_1 = [x + y for x, y in zip(turning_point_0, [0, 0.25])]
        turning_point_2 = [x + y for x, y in zip(turning_point_1, [-0.15, 0])]
        text_pos = [x + y for x, y in zip(turning_point_2, [-0.5, 0])]

        ax.annotate(
            "",
            xy=(x_axes, y_axes),
            xycoords=ax.transAxes,
            xytext=(turning_point_0[0], turning_point_0[1]),
            textcoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="-[,widthB=0.3,lengthB=0.3",
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
            "task context variables",
            xy=(text_pos[0], text_pos[1]),                  # arrow head (end)
            xycoords=ax.transAxes,
            xytext=(text_pos[0], text_pos[1]),        # arrow tail (start) 0.1 above
            textcoords=ax.transAxes,
            fontsize=font_size,
            color='black',
            va='center',
            clip_on=False
        )
    
    # # ---------------------------
    # # last layer xtick and action annotation
    # # ---------------------------
    # if i == len(layers_vis) - 1:
    #     # set xtick
    #     w = W.shape[1]

    #     ax.set_xticks([0, w - 1])
    #     ax.set_xticklabels([0, w - 1])

    #     # action annotation
    #     x_middle = 0.5
    #     y_bracket = -0.07
    #     y_text = -0.15

    #     ax.annotate(
    #         "actions",
    #         xy=(x_middle, y_bracket),
    #         xycoords=ax.transAxes,
    #         xytext=(x_middle, y_text),
    #         textcoords=ax.transAxes,
    #         ha="center",
    #         va="top",
    #         arrowprops=dict(
    #             arrowstyle="<-[,widthB=0.5,lengthB=0.3",
    #             color="black",
    #             linewidth=annotation_line_width,
    #         ),
    #         fontsize=font_size,
    #         color='black',
    #         clip_on=False
    #     )

# ---------------------------
# Aligh all subplots to bottom
# ---------------------------
bottom_y = 0.2

for ax in axes_list:
    pos = ax.get_position()
    height = pos.height
    ax.set_position([pos.x0, bottom_y, pos.width, height])

# ---------------------------
# Legend
# ---------------------------
legend_patches = [
    mpatches.Patch(color=shared_weights_color, label='shared weights across all tasks'),
    mpatches.Patch(color=partially_shared_weights_color, label='partially shared weights'),
]

for subtask in subtask_list:
    legend_patches.append(
        mpatches.Patch(color=subtask_colors[subtask], label=f'task {subtask} specific weights')
    )

legend_patches.append(
    mpatches.Patch(facecolor='white', edgecolor='grey', linewidth=0.8, label='unused weights')
)

fig.legend(
    handles=legend_patches,
    loc='lower center',
    bbox_to_anchor=(legend_pos[0], legend_pos[1]),
    ncol=1,
    fontsize=font_size,
    handlelength=1.0,
    handleheight=1.0,
)

fig.suptitle("subnetwork weights analysis across all tasks", fontsize=font_size)
fig.text(
    0.5,
    0.01,
    f"{plot_info}",
    ha='center',
    va='bottom',
    fontsize=font_size
)
# ---------------------------
# Save
# ---------------------------
save_fig = True
plot_name = "subnet_weights_analysis"
if save_fig:
    path = os.path.expanduser(f'~/Pictures/{experiment}_{plot_name}.pdf')
    plt.savefig(path)

# plt.show()