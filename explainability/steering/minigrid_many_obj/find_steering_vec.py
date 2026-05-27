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
# from rl_blox.blox.function_approximator.mlp import MLP
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
# (2) Get states for the sample (maybe unnecessary)
# ---------------------------

hparams_algorithm = dict(
    batch_size=1,
)

def sample_state(env, subtask=None, batch_size=128, key=jr.PRNGKey(seed)):
    # Get the observation space bounds
    low = jnp.array(env.observation_space.low)
    high = jnp.array(env.observation_space.high)

    # low and high are currently 0 and 255, but in env it is 0 and 6
    key, subkey = jr.split(key)

    # Sample a batch of states uniformly
    state = jax.random.randint(
        subkey,
        shape=(batch_size, low.shape[0]),
        minval=low,
        maxval=high,
    )

    obj_colors = OBJ_COLORS

    if subtask in obj_colors:
        # Sample only the subtask
        context = COLOR_TO_IDX[subtask]

    else:
        raise RuntimeError(f"Unknown subtask: {subtask}")

    # One-hot context
    one_hot_length = 6
    one_hot_context = jnp.zeros(one_hot_length)
    one_hot_context = one_hot_context.at[context].set(1)  # JAX-friendly

    # Broadcast the one-hot context to the batch
    # Replace the last 6 elements of every row in `state`
    state = state.at[:, -one_hot_length:].set(one_hot_context)

    return state


key, subkey = jr.split(key)
state = sample_state(env, subtask, hparams_algorithm.get("batch_size"), key=subkey)


# ---------------------------
# (3) Collect task activations
# ---------------------------
def collect_task_activations(task, policy, make_ocean_env, randomize_env=False):
    """
    Runs one episode in the environment and collects activations.

    Args:
        task: environment task name
        policy: function(obs) -> (action, activations)
        make_ocean_env: env constructor
        randomize_env: whether to randomize environment

    Returns:
        sum_reward: total episode reward
        all_activations: list of activations per timestep
    """
    sum_reward = 0.0
    eval_env = make_ocean_env(task, randomize=randomize_env, render_mode="human")

    obs, _ = eval_env.reset()
    done = False
    all_activations = []

    while not done:
        action, activations = policy(obs)

        all_activations.append(activations)

        obs, reward, terminated, truncated, info = eval_env.step(action)
        sum_reward += reward

        done = terminated or truncated

    eval_env.close()

    print(f"{task} reward: {sum_reward}")
    print(f"Collected {len(all_activations)} timesteps of activations")

    return sum_reward, all_activations

tasks = {"red", "grey"}

results = {}

for task in tasks:
    if task not in OBJ_COLORS:
        raise RuntimeError(f"Unknown task: {task}")

    reward, all_acts = collect_task_activations(
        task=task,
        policy=policy,
        make_ocean_env=make_ocean_env,
        randomize_env=randomize_env
    )

    results[task] = {
        "reward": reward,
        "activations": all_acts
    }

print(results["red"]["reward"])
print(results["grey"]["reward"])

red_activations = results["red"]["activations"]
grey_activations = results["grey"]["activations"]

# ---------------------------
# (4) Plot activations (timesteps × neuron activations per layer)
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

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5 * timesteps / 20),
        sharey=True
    )

    for ax, mat, name in zip(axes, matrices, layer_name):
        im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
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

plot_activation_layers(red_activations, title="Red Episode Activations")
plot_activation_layers(grey_activations, title="Grey Episode Activations")