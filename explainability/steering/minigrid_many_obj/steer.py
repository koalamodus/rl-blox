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

    def policy(obs, steer_vecs):
        q_values, activations = q([obs], steer_vecs)
        # print("q_values")
        # print(q_values)
        # print("activations")
        # print(activations)
        # print("--")

        return int(jnp.argmax(q_values)), activations

    return policy


policy = get_policy_from_q_net(q)

# ---------------------------
# (2) Load steering vector
# ---------------------------
tasks = {"red", "grey"}

load_data = True
if load_data:
    import pickle

    with open("data/minigrid_steering_vec.pkl", "rb") as file:
       first_steering_vec = pickle.load(file)
       second_steering_vec = pickle.load(file)

print(first_steering_vec)
print(second_steering_vec)

# ---------------------------
# (3) Activation steering (per layer)
# ---------------------------
# diff = red - grey

# red = diff + grey
# -> add steering vector to grey env, see if agent go to red corner
steer_vecs = jnp.stack([first_steering_vec, second_steering_vec])

def task_activations_steering(task, policy, steer_vecs, make_ocean_env, randomize_env=False):
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
        action, activations = policy(obs, steer_vecs)

        all_activations.append(activations)

        obs, reward, terminated, truncated, info = eval_env.step(action)
        sum_reward += reward

        done = terminated or truncated

    eval_env.close()

    print(f"{task} reward: {sum_reward}")
    print(f"Collected {len(all_activations)} timesteps of activations")

    return sum_reward, all_activations

tasks = {"grey"}

results = {}

for task in tasks:
    if task not in OBJ_COLORS:
        raise RuntimeError(f"Unknown task: {task}")

    reward, all_acts = task_activations_steering(
        task=task,
        policy=policy,
        steer_vecs=steer_vecs,
        make_ocean_env=make_ocean_env,
        randomize_env=randomize_env
    )

    results[task] = {
        "reward": reward,
        "activations": all_acts
    }

print(results["red"]["reward"])