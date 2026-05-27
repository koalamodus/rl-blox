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
task = "red"
# task = "blue"
# task = "purple"
# task = "grey"

if task not in OBJ_COLORS:
    raise RuntimeError(f"Unknown task: {task}")

sum_reward = 0.0
eval_env = make_ocean_env(task, randomize=randomize_env, render_mode="human")
obs, _ = eval_env.reset()
done = False
while not done:
    action, activations = policy(obs)
    print(f"activations: {activations}")

    obs, reward, terminated, truncated, info = eval_env.step(action)
    sum_reward += reward
    if terminated or truncated:
        print(f"{task} reward: {reward}")
        print(f"{task} return: {sum_reward}")
        eval_env.close()
        done = True