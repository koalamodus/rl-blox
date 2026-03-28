import os
from holoocean_envs import sample_context
from holoocean_envs import make_holoocean_env
from holoocean_envs import TASK_NAMES, TASK_TO_IDX
import flax.nnx as nnx
import numpy as np

seed = 49  # random seed for np and jax
OBJ_COLORS = TASK_NAMES
subtasks = TASK_NAMES

# ---------------------------
# Load trained or pruned network
# ---------------------------
import orbax.checkpoint as ocp
from tqdm import tqdm
from rl_blox.blox.function_approximator.mlp import MLP

# Choose ckpt from pre-trained full q-net
experiment = "holoocean"
randomize_env = True
seed_num = 0
file_name = "_minigrid_holoocean_medium_DDQN-UTS_1774215135.9690797_q_step_000250000_epoch_250000"
model_hidden_nodes = [128, 128]

eval_repetition = 20 if randomize_env else 1

# Define evaluation function
def eval_policy(task, env, policy, current=None, verbose=False, repetition=1, seed=None):
    task_score_info = {}
    sum_reward = 0.0
    sum_success_rate = 0.0

    rng = np.random.default_rng(seed=seed)
    env_seeds = rng.integers(10000, size=repetition).tolist()

    num_episode = repetition
    
    # randomize both current and organisms position with seeds
    # same set of current and organisms are used for evaluate all networks, when seed is the same
    for eps in tqdm(range(repetition)):
        context = sample_context(task, current, seed=env_seeds[eps])
        # print(f"context: {context}")

        obs, _ = env.reset(context, seed=env_seeds[eps])
        # print(f"obs: {obs}")
        # print(f"{env.red_organisms[7]}, {env.black_organisms[9]}")
        terminated = truncated = False

        while not (terminated or truncated):
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            sum_reward += reward
        
        task_context = TASK_TO_IDX[task]
        sum_success_rate += obs[task_context-4]
        print(f"obs: {obs}")

    task_score_info.update({
        f"Task {task}": {
            "Avg Return": sum_reward / num_episode,
            "Success Rate": sum_success_rate / num_episode
        }
    })
    if verbose:
        print(task_score_info)
    return task_score_info

def evaluate_policy_on_task(env, policy, tasks=None, obj_colors=OBJ_COLORS, current=None, repetition=eval_repetition, render_mode=None, seed=None, verbose=True):
    if tasks == None:
        tasks = obj_colors
    else:
        tasks = tasks if isinstance(tasks, list) else [tasks]
    
    all_task_scores = {}  # store results for all colors
    eval_env = env

    for task in tasks:
        assert task in TASK_NAMES
        
        task_score_info = eval_policy(task, eval_env, policy, current, verbose, repetition, seed)
        all_task_scores.update(task_score_info)
    return all_task_scores

def get_policy_from_q_net(q):

    def policy(obs):
        return int(np.argmax(q([obs])))

    return policy

# Set up environment to get input/output shapes
subtask = subtasks[0]
env = make_holoocean_env(render_mode=None, contextual=True)

step = int(file_name.split("_")[-3])
ckpt_subnet = os.path.expanduser(
    f"~/workspace/XRL/ocean_subnet/{experiment}/UTS/seed_{seed_num}/step_{step}/subnetwork_{subtask}"
)

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

# Set up ckpt path
ckpt_full_net = os.path.expanduser(
    f"~/workspace/XRL/ocean_trained_model/{experiment}/UTS/seed_{seed_num}/{file_name}"
)
restored_model_full = checkpointer.restore(ckpt_full_net, abstract_state)
q_full = nnx.merge(graphdef, restored_model_full)
policy_full = get_policy_from_q_net(q_full)

print("Evaluate full network")
full_scores = evaluate_policy_on_task(env, policy_full, tasks=subtasks, repetition=eval_repetition, render_mode=None,seed=seed)

print("Full network evaluation complete.")

# ---------------------------
# Evaluate subnetworks
# ---------------------------
subnet_scores = {}
performance_drop = {}

for subtask in subtasks:
    step = int(file_name.split("_")[-3])
    ckpt_subnet = os.path.expanduser(
        f"~/workspace/XRL/ocean_subnet/{experiment}/UTS/seed_{seed_num}/step_{step}/subnetwork_{subtask}"
    )

    restored_model_sub = checkpointer.restore(ckpt_subnet, abstract_state)
    q_sub = nnx.merge(graphdef, restored_model_sub)
    policy_sub = get_policy_from_q_net(q_sub)

    print(f"Evaluate subnetwork {subtask}")
    scores = evaluate_policy_on_task(env, policy_sub, tasks=subtasks, repetition=eval_repetition, seed=seed)
    subnet_scores[subtask] = scores

    # Compute performance drop relative to full network
    drop = {
        f"Task {t}": {
            metric: scores[f"Task {t}"][metric] / full_scores[f"Task {t}"][metric] 
            for metric in full_scores[f"Task {t}"]
        }
        for t in subtasks
    }
    performance_drop[subtask] = drop

print("Subnetwork evaluation complete.")
print("Performance drop compared to full network:")
for subtask, drops in performance_drop.items():
    print(f"subnetwork {subtask}: {drops}")

env.close()
# ---------------------------
# Save evaluation result
# ---------------------------

import pickle

eval_results = {
    "full_scores": full_scores,
    "subnet_scores": subnet_scores,
    "performance_drop": performance_drop
}

with open("evaluation_results.pkl", "wb") as f:
    pickle.dump(eval_results, f)

print("All evaluation results saved in 'evaluation_results.pkl'.")