import os
from find_many_objects_env import make_ocean_env

import flax.nnx as nnx
import numpy as np

seed = 49  # random seed for np and jax
plot_mask_only = True

# ---------------------------
# (1) Load trained network
# ---------------------------
import orbax.checkpoint as ocp
from rl_blox.blox.function_approximator.mlp import MLP

# Choose ckpt from pre-trained full q-net

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
ckpt_full_net = os.path.expanduser(
    f"~/workspace/XRL/ocean_trained_model/{experiment}/UTS/seed_{seed_num}/{file_name}"
)
step = int(file_name.split("_")[-3])
plot_info = f"{experiment}, seed {seed_num}, step {step}"

# Set up environment to get input/output shapes
subtask = "purple"
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

#--------------

subtasks = ["purple", "blue", "grey", "red"]
weights_dict = {}

for subtask in subtasks:
    env = make_ocean_env(subtask)
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

custom_cmap.set_bad(color='black')  # Black for masked (zero in all subtasks)

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
fig.suptitle(f"zero weights in all subtasks", fontsize=14)
fig.text(
    0.5,
    0.01,
    f"{plot_info}",
    ha='center',
    va='bottom',
    fontsize=8
)

# Legend for masked weights
masked_patch = mpatches.Patch(color='black', label='Zero in all subtasks')
fig.legend(handles=[masked_patch], loc='lower center', bbox_to_anchor=(0.7, 0.10),
           ncol=1, fontsize=8)

fig.suptitle(f"Masked weights across all subtasks", fontsize=14)
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

    for i in range(num_layers):
        stacked = np.stack([weights_dict[sub][i] for sub in subtask_list], axis=0)
        zero_all_mask = (stacked == 0).all(axis=0)  # grey
        current_weights = stacked[subtask_list.index(subtask)]
        black_mask = (current_weights == 0) & (~zero_all_mask)  # Black

        # Start with weights
        plot_array = np.copy(current_weights)
        
        # Use masked arrays for special coloring
        masked_array = np.ma.masked_array(plot_array)
        # Set masks
        masked_array.mask = False  # start with no mask

        # We will override colors in imshow using a colormap with "bad" colors
        # We'll map grey and black manually via masked array trick
        masked_array = np.ma.array(plot_array, mask=zero_all_mask)  # mask global zeros

        masked_layers.append((masked_array, black_mask, zero_all_mask))
        layer_name = "Output Layer" if i == num_layers - 1 else f"Hidden Layer {i}"
        titles.append(layer_name)

    # Plot
    fig, axes = plt.subplots(1, num_layers, figsize=(num_layers*3, 3))
    if num_layers == 1:
        axes = [axes]

    for ax, (W_masked, black_mask, grey_mask), title in zip(axes, masked_layers, titles):
        # Base colormap for normal weights
        im = ax.imshow(W_masked, cmap='viridis', aspect='equal')
        
        # Overlay grey for global zeros
        grey_overlay = np.zeros_like(W_masked, dtype=float)
        grey_overlay[grey_mask] = 1  # any non-zero to show
        ax.imshow(grey_overlay, cmap='Greys', alpha=0.5)
        
        # Overlay black for subtask zeros
        black_overlay = np.zeros_like(W_masked, dtype=float)
        black_overlay[black_mask] = 1
        ax.imshow(black_overlay, cmap='Greys', alpha=1.0)
        
        ax.set_title(title)
        ax.set_xlabel("Output Neuron")
        ax.set_ylabel("Input Neuron")

    # Legend
    patches = [
        mpatches.Patch(color='grey', label='Zero across all subtasks'),
        mpatches.Patch(color='black', label='Zero in this subtask only')
    ]
    fig.legend(handles=patches, loc='lower center', ncol=2)
    fig.suptitle(f"Subtask: {subtask}", fontsize=14)
    plt.tight_layout()
    plt.show()