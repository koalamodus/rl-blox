import os
from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX
from find_many_objects_env import make_ocean_env

import jax
import flax.nnx as nnx
import jax.numpy as jnp
import jax.random as jr

seed = 49  # random seed for np and jax
key = jr.PRNGKey(seed)

# ---------------------------
# (1) Load trained network
# ---------------------------
import orbax.checkpoint as ocp
from mlp_for_steering import MLP

# Choose ckpt

experiment = "reefshield_random_medium"
randomize_env = True
seed_num = 0
file_name = "_minigrid_reefshield_random_medium_DDQN-UTS_1773787402.9864454_q_step_005000000_epoch_5000000"
model_hidden_nodes = [32, 32]

# Set up ckpt path
ckpt_path = os.path.expanduser(
    f"~/workspace/xrl/ocean_trained_model/{experiment}/UTS/seed_{seed_num}/{file_name}"
)

# Set up environment to get input/output shapes
OBJ_COLORS = COLOR_NAMES
subtask = "red"
env_name = f"ocean_{subtask}_{experiment}"
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
restored_model = checkpointer.restore(ckpt_path, abstract_state)
q = nnx.merge(graphdef, restored_model)

# wrap model to capture intermediates
q = nnx.capture(q, nnx.Intermediate)

print("Loaded model checkpoint.")


def get_policy_from_q_net(q):

    def policy(obs):
        q_values, activations = q([obs])
        # print("q_values")
        # print(q_values)
        # print("activations")
        # print(activations)
        # print("--")

        return int(jnp.argmax(q_values)), activations

    return policy


policy = get_policy_from_q_net(q)

# ---------------------------
# (2) Collect task activations
# ---------------------------
tasks = {"red", "grey"}

load_data = True
if load_data:
    import pickle

    with open("data/minigrid_activations.pkl", "rb") as file:
       red_activations = pickle.load(file)
       grey_activations = pickle.load(file)

print(len(red_activations))
print(len(grey_activations))

# ---------------------------
# (3) Plot activations (timesteps × neuron activations per layer)
# ---------------------------

def get_layer_activation_matrix(task_episode_activations, layer_idx=0, name="layer", verbose=False):
    """
    Extracts (timesteps, neurons) matrix for a given layer index.

    task_episode_activations: list of steps, each step["hidden"] shape (batch, layers, neurons)
    layer_idx: which layer to extract (0 = first layer, 1 = second layer, etc.)
    """
    matrix = jnp.stack([
        step["hidden"][0][layer_idx] for step in task_episode_activations
    ])
    matrix = matrix[:, 0, :]  # (timesteps, neurons), for batch size 1

    if verbose:
        print(f"{name} activation matrix shape: {matrix.shape}")
        print(matrix)

    return matrix

import matplotlib.pyplot as plt

def plot_activation_layers(
    task_episode_activations,
    layer_idx=(0, 1),
    layer_name=("First Layer", "Second Layer"),
    title="Task Episode Activations"
):
    """
    Extract and plot two activation layers side-by-side with shared color scale.
    """

    timesteps = len(task_episode_activations)

    # extract both layers
    matrices = [
        get_layer_activation_matrix(task_episode_activations, idx, name)
        for idx, name in zip(layer_idx, layer_name)
    ]

    first, second = matrices

    # shared color scale
    vmin = min(first.min(), second.min())
    vmax = max(first.max(), second.max())

    absmax = max(abs(vmax), abs(vmin))

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5 * timesteps / 20),
        sharey=True
    )

    for ax, mat, name in zip(axes, matrices, layer_name):
        im = ax.imshow(mat, aspect='auto', cmap='bwr', vmin=-absmax, vmax=absmax)
        ax.set_title(name)
        ax.set_xlabel("Neurons")
        ax.set_yticks(jnp.arange(0, mat.shape[0], 5))

    axes[0].set_ylabel("Timesteps")

    fig.suptitle(title)

    # single shared colorbar (attached to second subplot)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Activation value")

    plt.tight_layout()
    plt.show()

    return first, second

red_first, red_second = plot_activation_layers(red_activations, title="Red Episode Activations")
grey_first, grey_second = plot_activation_layers(grey_activations, title="Grey Episode Activations")



# ---------------------------
# (4) Get avg activations diff (neuron activations per layer)
# ---------------------------
# diff = red - grey


# take first n timestamps
n = 5
diff_first = red_first[0:n] - grey_first[0:n]
diff_second = red_second[0:n] - grey_second[0:n]

avg_first_diff = jnp.mean(diff_first, axis=0)
avg_second_diff = jnp.mean(diff_second, axis=0)

print(avg_first_diff)
print(avg_second_diff)

def plot_diff_matrix(
    first,
    second,
    timesteps,
    layer_name=("First Layer", "Second Layer"),
    title="Task Activation Difference"
):
    # shared color scale
    vmin = min(first.min(), second.min())
    vmax = max(first.max(), second.max())
    absmax = max(abs(vmax), abs(vmin))

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5 * timesteps / 20 + 1),
        sharey=True
    )

    matrices = [first, second]

    for ax, mat, name in zip(axes, matrices, layer_name):
        im = ax.imshow(mat, aspect='auto', cmap='bwr', vmin=-absmax, vmax=absmax)
        ax.set_title(name)
        ax.set_xlabel("Neurons")
        ax.set_yticks(jnp.arange(0, mat.shape[0], 1))

    axes[0].set_ylabel("Timesteps")

    fig.suptitle(title)

    # single shared colorbar (attached to second subplot)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Activation value")

    plt.tight_layout()
    plt.show()

plot_diff_matrix(diff_first, diff_second, n,
    title="Activation Difference (red - grey)"
)

plot_diff_matrix(avg_first_diff[None, :], avg_second_diff[None, :], 1,
    title="Average Activation Difference (red - grey)"
)


# ---------------------------
# (5) Get top-k activations diff (in neuron activations per layer)
# ---------------------------

def topk_abs_values(x, k=7):
    top_idx = jnp.argsort(-jnp.abs(x))[:k]
    top_values = x[top_idx]
    return top_idx, top_values

top_idx_1, top_values_1 = topk_abs_values(avg_first_diff)
top_idx_2, top_values_2 = topk_abs_values(avg_second_diff)

print(list(zip(top_idx_1.tolist(), top_values_1.tolist())))
print(list(zip(top_idx_2.tolist(), top_values_2.tolist())))

# ---------------------------
# (6) Get steering vector (in neuron activations per layer)
# ---------------------------

def get_steering_vec(top_idx, top_values, vec_length=32):
    vec = jnp.zeros(vec_length)
    vec = vec.at[top_idx].set(top_values)
    return vec

first_steering_vec = get_steering_vec(top_idx_1, top_values_1)
second_steering_vec = get_steering_vec(top_idx_2, top_values_2)

print(first_steering_vec)
print(second_steering_vec)

save_data = True
if save_data:
    import pickle

    with open("data/minigrid_steering_vec.pkl", "wb") as file:
        pickle.dump(first_steering_vec, file)
        pickle.dump(second_steering_vec, file)

# ---------------------------
# (7) Activation steering (per layer)
# ---------------------------

# diff = red - grey

# red = diff + grey
# -> add steering vector to grey env, see if agent go to red corner

