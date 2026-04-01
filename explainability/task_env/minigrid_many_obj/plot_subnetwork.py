import os
from find_many_objects_env import make_ocean_env

import flax.nnx as nnx
import jax.random as jr

seed = 49  # random seed for np and jax
key = jr.PRNGKey(seed)

# ---------------------------
# (1) Load trained network
# ---------------------------
import orbax.checkpoint as ocp
from rl_blox.blox.function_approximator.mlp import MLP
from minigrid.core.constants import COLOR_NAMES
subtasks = COLOR_NAMES

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

# Set up environment to get input/output shapes
subtask = subtasks[0]
env = make_ocean_env(subtask)

step = int(file_name.split("_")[-3])
ckpt_subnet = os.path.expanduser(
    f"~/workspace/XRL/ocean_subnet/{experiment}/UTS/seed_{seed_num}/step_{step}/subnetwork_{subtask}"
)

# choose ckpt path
task, ckpt_path = f"Pretrained network", ckpt_full_net
# task, ckpt_path = f"Subnetwork {subtask}", ckpt_subnet

plot_info = f"{experiment}, seed {seed_num}, step {step}"

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
q_net = nnx.merge(graphdef, restored_model)

print("Loaded model checkpoint.")

# ---------------------------
# Plot MLP weights - black is 0
# ---------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

print("Plotting network weights (aligned axes & equal subplot sizes)...")

# Collect weight matrices
weights = []
titles = []

for i, layer in enumerate(q_net.hidden_layers):
    W = np.array(layer.kernel.value)
    weights.append(W)
    titles.append(f"Hidden Layer {i}")

W_out = np.array(q_net.output_layer.kernel.value)
weights.append(W_out)
titles.append("Output Layer")

# Shared color scale
abs_max = max(abs(W).max() for W in weights)
norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

# Custom diverging colormap: white at 0
colors = [
    (0, "blue"),
    (0.49, "lightblue"),
    (0.5, "white"),
    (0.51, "lightcoral"),
    (1, "red")
]
custom_cmap = LinearSegmentedColormap.from_list("highlight_zero", colors)
# Set masked values (zeros) to black
custom_cmap.set_bad(color='black')

# Scale subplot sizes by matrix width
margin_width_ratio = 1.5
width_ratios = [W.shape[1] for W in weights] + [margin_width_ratio]  # last is colorbar

# Scale figure size based on total width
scale = 0.1  # tweak this to control overall size
fig_width = sum(width_ratios) * scale
fig_height = max(W.shape[0] for W in weights) * scale

fig = plt.figure(figsize=(fig_width, fig_height))

# Gridspec
gs = fig.add_gridspec(
    1,
    len(weights) + 1,
    width_ratios=width_ratios,
    wspace=0.5  # tweak this to control horizontal space between layers
)

axes = []
for i, W in enumerate(weights):
    ax = fig.add_subplot(gs[0, i])
    
    # Mask exact zeros
    masked_W = np.ma.masked_equal(W, 0.0)
    
    # Plot with square cells
    im = ax.imshow(masked_W, cmap=custom_cmap, norm=norm, aspect='equal')
    
    # Labels
    if i == 0:
        ax.set_xlabel("Output Neuron Index", fontsize=10)
        ax.set_ylabel("Input Neuron Index", fontsize=10)
    
    # Per-layer limits
    h, w = W.shape
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    
    # Output layer ticks only at first layer
    if i == len(weights) - 1:
        ax.set_xticks([0, w - 1])
        ax.set_xticklabels([0, w - 1])
    
    ax.set_title(titles[i], fontsize=10)
    ax.tick_params(axis='both', labelsize=8)  # set tick label size
    axes.append(ax)

# Colorbar
cax = fig.add_subplot(gs[0, -1])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Weight Value", fontsize=10)
cbar.ax.tick_params(labelsize=8)

# Figure title and bottom text
fig.suptitle(f"{task} weights", fontsize=14)
fig.text(
    0.5,      # x-coordinate (center)
    0.01,     # y-coordinate (near bottom)
    f"{plot_info}",
    ha='center',
    va='bottom',
    fontsize=8
)

import matplotlib.patches as mpatches
# Add black grid legend for masked values
masked_patch = mpatches.Patch(color='black', label='Masked (zero) weight')
fig.legend(handles=[masked_patch], loc='lower center', bbox_to_anchor=(0.7, 0.10),
           ncol=1, fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
