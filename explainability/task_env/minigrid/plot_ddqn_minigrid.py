import pickle
import jax.numpy as jnp
from flax import nnx
from minigrid.core.constants import COLOR_NAMES
from minigrid_one_context_env import make_ocean_env

# ---------------------------
# Load trained Q-network
# ---------------------------
print("Evaluate trained model.")

ckpt_path = "ddqn_minigrid_ckpt.pkl"
# ckpt_path = "ddqn_subnet_None_ckpt.pkl"

with open(ckpt_path, "rb") as f:
    mlp_state = pickle.load(f)

print("Loaded model checkpoint.")

# Recreate the same MLP architecture
from rl_blox.blox.function_approximator.mlp import MLP

# Set up environment to get input/output shapes
obj_colors = ["red", "green", "blue"]
subtask = None  # or a specific color
env = make_ocean_env(target_color=subtask, obj_colors=obj_colors)

q_net = MLP(
    env.observation_space.shape[0],
    int(env.action_space.n),
    rngs=nnx.Rngs(42),  # seed doesn't matter here
    activation="relu",
    hidden_nodes=[128, 128, 128],
)

# Restore parameters from checkpoint
nnx.update(q_net, mlp_state)

# Define policy function
def get_policy_from_q_net(q_net):
    def policy(obs):
        return int(jnp.argmax(q_net([obs])))
    return policy

policy = get_policy_from_q_net(q_net)

# ---------------------------
# Evaluate the policy
# ---------------------------

from eval_helper import eval_policy

def evaluate_policy_on_task(policy, task=None, obj_colors=obj_colors, render_mode="human"):
    if task is None:
        task = obj_colors
    else:
        task = task if isinstance(task, list) else [task]

    for target_color in task:
        assert target_color in COLOR_NAMES
        eval_env = make_ocean_env(target_color, obj_colors, render_mode)
        _ = eval_policy(target_color, eval_env, policy, verbose=True, num_episode=20)
        eval_env.close()

# Example evaluation
print("Evaluating loaded Q-network")
evaluate_policy_on_task(policy, task=subtask, obj_colors=obj_colors, render_mode="None")

# ---------------------------
# Run interactive demo
# ---------------------------

# eval_env = make_ocean_env(target_color=subtask, obj_colors=obj_colors, render_mode="human")
# obs, _ = eval_env.reset()
# while True:
#     action = policy(obs)
#     obs, reward, terminated, truncated, info = eval_env.step(action)
#     if terminated or truncated:
#         print(f"{eval_env.current_task_color} reward: {reward}")
#         obs, _ = eval_env.reset()

# ---------------------------
# Plot MLP weights - white is 0
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

# Determine maximum input and output neuron indices across all layers
max_input = max(W.shape[1] for W in weights)
max_output = max(W.shape[0] for W in weights)

# Figure size: make each subplot proportional to max neurons
subplot_width = 4
subplot_height = 4
fig = plt.figure(figsize=(subplot_width * len(weights), subplot_height))

# Gridspec with colorbar
gs = fig.add_gridspec(1, len(weights) + 1, width_ratios=[1]*len(weights) + [0.05], wspace=0.3)

axes = []
for i, W in enumerate(weights):
    ax = fig.add_subplot(gs[0, i])
    
    # Mask exact zeros
    masked_W = np.ma.masked_equal(W, 0.0)
    
    # Plot with masked zeros and custom colormap
    im = ax.imshow(masked_W, cmap=custom_cmap, norm=norm, aspect='equal')
    
    # Labels
    ax.set_xlabel("Output Neuron Index")
    if i == 0:
        ax.set_ylabel("Input Neuron Index")
    else:
        ax.set_yticklabels([])
    
    # Fix axis limits to align all subplots
    ax.set_xlim(-0.5, max_input-0.5)
    ax.set_ylim(max_output-0.5, -0.5)  # invert y-axis to match imshow
    
    # Output layer x-ticks only at first and last neuron
    if i == len(weights) - 1:
        ax.set_xticks([0, W.shape[1]-1])
        ax.set_xticklabels([0, W.shape[1]-1])
    
    ax.set_title(titles[i])
    axes.append(ax)

# Colorbar next to last layer
cax = fig.add_subplot(gs[0, -1])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Weight Value")

# Figure-wide title
fig.suptitle(f"Network Weights: {ckpt_path}", fontsize=16)


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# print("Plotting network weights (highlight zeros)...")

# # Collect weight matrices
# weights = []
# titles = []

# for i, layer in enumerate(q_net.hidden_layers):
#     W = np.array(layer.kernel.value)
#     weights.append(W)
#     titles.append(f"Hidden Layer {i}")

# W_out = np.array(q_net.output_layer.kernel.value)
# weights.append(W_out)
# titles.append("Output Layer")

# # Shared color scale
# abs_max = max(abs(W).max() for W in weights)
# norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

# # Custom diverging colormap: white at 0
# colors = [
#     (0, "blue"),       
#     (0.49, "lightblue"),
#     (0.5, "white"),    
#     (0.51, "lightcoral"),
#     (1, "red")         
# ]
# custom_cmap = LinearSegmentedColormap.from_list("highlight_zero", colors)

# # Create figure with gridspec
# fig = plt.figure(figsize=(15, 5))
# gs = fig.add_gridspec(1, len(weights) + 1, width_ratios=[1]*len(weights) + [0.05], wspace=0.2)

# axes = []
# for i, W in enumerate(weights):
#     ax = fig.add_subplot(gs[0, i])
#     im = ax.imshow(W, cmap=custom_cmap, norm=norm, aspect="equal")
    
#     # Swap x and y labels
#     ax.set_xlabel("Output Neuron Index")
    
#     # Only show y-label on the first subplot to prevent overlap
#     if i == 0:
#         ax.set_ylabel("Input Neuron Index")
#     else:
#         ax.set_yticklabels([])

#     # Customize x-axis for output layer
#     if i == len(weights) - 1:  # last subplot = output layer
#         ax.set_xticks([0, W.shape[1]-1])
#         ax.set_xticklabels([0, W.shape[1]-1])
    
#     ax.set_title(titles[i])
#     axes.append(ax)

# # Colorbar next to output layer
# cax = fig.add_subplot(gs[0, -1])
# cbar = fig.colorbar(im, cax=cax)
# cbar.set_label("Weight Value")

# # Figure-wide title
# fig.suptitle(f"Network Weights: {ckpt_path}", fontsize=16)

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# print("Plotting network weights (highlight zeros)...")

# # Collect weight matrices
# weights = []
# titles = []

# for i, layer in enumerate(q_net.hidden_layers):
#     W = np.array(layer.kernel.value)
#     weights.append(W)
#     titles.append(f"Hidden Layer {i}")

# W_out = np.array(q_net.output_layer.kernel.value)
# weights.append(W_out)
# titles.append("Output Layer")

# # Shared color scale
# abs_max = max(abs(W).max() for W in weights)
# norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

# # Custom diverging colormap: white at 0
# colors = [
#     (0, "blue"),       # negative extreme
#     (0.49, "lightblue"),
#     (0.5, "white"),    # exact zero
#     (0.51, "lightcoral"),
#     (1, "red")         # positive extreme
# ]
# custom_cmap = LinearSegmentedColormap.from_list("highlight_zero", colors)

# # Create figure with gridspec
# fig = plt.figure(figsize=(15, 5))
# gs = fig.add_gridspec(1, len(weights) + 1, width_ratios=[1]*len(weights) + [0.05], wspace=0.1)

# axes = []
# for i in range(len(weights)):
#     ax = fig.add_subplot(gs[0, i])
#     im = ax.imshow(weights[i], cmap=custom_cmap, norm=norm, aspect="equal")
    
#     # Swap x and y labels
#     ax.set_xlabel("Output Neuron Index")
#     ax.set_ylabel("Input Neuron Index")
    
#     ax.set_title(titles[i])
#     axes.append(ax)

# # Colorbar next to output layer
# cax = fig.add_subplot(gs[0, -1])
# cbar = fig.colorbar(im, cax=cax)
# cbar.set_label("Weight Value")

# # Add a figure-wide title
# fig.suptitle("MLP Network Weights (Zeros in White)", fontsize=16)

# plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# print("Plotting network weights (highlight zeros)...")

# # Collect weight matrices
# weights = []
# titles = []

# for i, layer in enumerate(q_net.hidden_layers):
#     W = np.array(layer.kernel.value)
#     weights.append(W)
#     titles.append(f"Hidden Layer {i}")

# W_out = np.array(q_net.output_layer.kernel.value)
# weights.append(W_out)
# titles.append("Output Layer")

# # Shared color scale
# abs_max = max(abs(W).max() for W in weights)
# norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

# # Custom diverging colormap: white at 0, rapid color change near 0
# colors = [
#     (0, "blue"),   # negative extreme
#     (0.49, "lightblue"),
#     (0.5, "white"),  # exact zero
#     (0.51, "lightcoral"),
#     (1, "red")     # positive extreme
# ]
# custom_cmap = LinearSegmentedColormap.from_list("highlight_zero", colors)

# # Create figure with gridspec
# fig = plt.figure(figsize=(15, 5))
# gs = fig.add_gridspec(1, len(weights) + 1, width_ratios=[1]*len(weights) + [0.05], wspace=0.1)

# axes = []
# for i in range(len(weights)):
#     ax = fig.add_subplot(gs[0, i])
#     im = ax.imshow(weights[i], cmap=custom_cmap, norm=norm, aspect="equal")
#     ax.set_title(titles[i])
#     ax.set_xlabel("Input Neuron Index")
#     ax.set_ylabel("Output Neuron Index")
#     axes.append(ax)

# # Colorbar next to output layer
# cax = fig.add_subplot(gs[0, -1])
# cbar = fig.colorbar(im, cax=cax)
# cbar.set_label("Weight Value")

# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import TwoSlopeNorm

# print("Plotting network weights (zero = white)...")

# # Collect weight matrices
# weights = []
# titles = []

# for i, layer in enumerate(q_net.hidden_layers):
#     W = np.array(layer.kernel.value)
#     weights.append(W)
#     titles.append(f"Hidden Layer {i}")

# W_out = np.array(q_net.output_layer.kernel.value)
# weights.append(W_out)
# titles.append("Output Layer")

# # Determine shared scale and zero-centered norm
# abs_max = max(abs(W).max() for W in weights)
# norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

# # Use gridspec for tight control
# fig = plt.figure(figsize=(15, 5))
# gs = fig.add_gridspec(1, len(weights) + 1, width_ratios=[1]*len(weights) + [0.05], wspace=0.1)

# axes = []
# for i in range(len(weights)):
#     ax = fig.add_subplot(gs[0, i])
#     im = ax.imshow(weights[i], cmap="seismic", norm=norm, aspect="equal")
#     ax.set_title(titles[i])
#     ax.set_xlabel("Input Neuron Index")
#     ax.set_ylabel("Output Neuron Index")
#     axes.append(ax)

# # Colorbar next to output layer
# cax = fig.add_subplot(gs[0, -1])
# cbar = fig.colorbar(im, cax=cax)
# cbar.set_label("Weight Value")

# plt.tight_layout()
# plt.show()

# ---------------------------
# Plot MLP weights
# ---------------------------

# import numpy as np
# import matplotlib.pyplot as plt

# print("Plotting network weights...")

# # Collect weight matrices
# weights = []
# titles = []

# for i, layer in enumerate(q_net.hidden_layers):
#     W = np.array(layer.kernel.value)
#     weights.append(W)
#     titles.append(f"Hidden Layer {i}")

# W_out = np.array(q_net.output_layer.kernel.value)
# weights.append(W_out)
# titles.append("Output Layer")

# # Shared color scale
# vmin = min(W.min() for W in weights)
# vmax = max(W.max() for W in weights)

# # Use gridspec to control spacing
# fig = plt.figure(figsize=(15, 5))
# gs = fig.add_gridspec(1, len(weights) + 1, width_ratios=[1]*(len(weights)) + [0.05], wspace=0.1)

# axes = []
# for i in range(len(weights)):
#     ax = fig.add_subplot(gs[0, i])
#     im = ax.imshow(weights[i], cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
#     ax.set_title(titles[i])
#     ax.set_xlabel("Input Neuron Index")
#     ax.set_ylabel("Output Neuron Index")
#     axes.append(ax)

# # Colorbar only next to the output layer
# cax = fig.add_subplot(gs[0, -1])
# cbar = fig.colorbar(im, cax=cax)
# cbar.set_label("Weight Value")

# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# print("Plotting network weights...")

# # Collect weight matrices
# weights = []
# titles = []

# for i, layer in enumerate(q_net.hidden_layers):
#     W = np.array(layer.kernel.value)
#     weights.append(W)
#     titles.append(f"Hidden Layer {i}")

# W_out = np.array(q_net.output_layer.kernel.value)
# weights.append(W_out)
# titles.append("Output Layer")

# # Use shared color scale
# vmin = min(W.min() for W in weights)
# vmax = max(W.max() for W in weights)

# # Create subplots
# fig, axes = plt.subplots(1, len(weights), figsize=(15, 5))

# for ax, W, title in zip(axes, weights, titles):
#     im = ax.imshow(W, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
#     ax.set_title(title)
#     ax.set_xlabel("Input Neuron Index")
#     ax.set_ylabel("Output Neuron Index")

# # Colorbar only to the right of the last subplot
# cbar = fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
# cbar.set_label("Weight Value")

# plt.tight_layout()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# print("Plotting network weights...")

# # Collect weight matrices
# weights = []
# titles = []

# for i, layer in enumerate(q_net.hidden_layers):
#     W = np.array(layer.kernel.value)
#     weights.append(W)
#     titles.append(f"Hidden Layer {i}")

# W_out = np.array(q_net.output_layer.kernel.value)
# weights.append(W_out)
# titles.append("Output Layer")

# # Use shared color scale
# vmin = min(W.min() for W in weights)
# vmax = max(W.max() for W in weights)

# # Create subplots
# fig, axes = plt.subplots(1, len(weights), figsize=(15, 5))

# for ax, W, title in zip(axes, weights, titles):
#     im = ax.imshow(W, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
#     ax.set_title(title)
#     ax.set_xlabel("Input Neuron Index")
#     ax.set_ylabel("Output Neuron Index")

# # Single colorbar for all plots
# cbar = fig.colorbar(im, ax=axes, shrink=0.8)
# cbar.set_label("Weight Value")

# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# def plot_weights(weights, cmap="viridis", vmin=None, vmax=None, title=None):
#     fig, ax = plt.subplots(figsize=(6, 6))

#     im = ax.imshow(weights, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

#     # Axis labels
#     ax.set_xlabel("Input Neuron Index")
#     ax.set_ylabel("Output Neuron Index")

#     # Title
#     if title is not None:
#         ax.set_title(title)

#     # Colorbar
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label("Weight Value")

#     plt.tight_layout()
#     plt.show()


# print("Plotting network weights...")

# # Hidden layers
# for i, layer in enumerate(q_net.hidden_layers):
#     W = np.array(layer.kernel.value)   # convert JAX -> numpy
#     plot_weights(W, title=f"Hidden Layer {i} Weight Matrix")

# # Output layer
# W_out = np.array(q_net.output_layer.kernel.value)
# plot_weights(W_out, title="Output Layer Weight Matrix")

# import numpy as np
# import matplotlib.pyplot as plt

# def plot_tight(weights, cmap="viridis", vmin=None, vmax=None, title=None):
#     fig, ax = plt.subplots(figsize=(4,4))
#     ax.imshow(weights, cmap=cmap, vmin=vmin, vmax=vmax)
#     ax.axis("off")
#     if title is not None:
#         ax.set_title(title)
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.show()


# print("Plotting network weights...")

# # Hidden layers
# for i, layer in enumerate(q_net.hidden_layers):
#     W = np.array(layer.kernel.value)   # convert JAX -> numpy
#     plot_tight(W, title=f"Hidden Layer {i} Weights")

# # Output layer
# W_out = np.array(q_net.output_layer.kernel.value)
# plot_tight(W_out, title="Output Layer Weights")