import os

import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp
from rl_blox.blox.function_approximator.mlp import MLP


def eval_policy(color, env, policy, verbose=False, num_episode=1):
    task_score_info = {}
    sum_ep_reward = 0.0
    num_task_success = 0.0
    for _ in range(num_episode):
        ep_reward = 0.0
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action = policy(obs)
            # print(f"action: {action}")
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if reward > 0.0:
                num_task_success += 1
            sum_ep_reward += ep_reward

    task_score_info.update({
        f"Task {color}": {
            "Return": sum_ep_reward / num_episode,
            "Success Rate": num_task_success / num_episode
        }
    })
    # task_score_info.update({f"Task {color}": ep_reward})

    if verbose:
        print(task_score_info)

    return task_score_info


def get_policy_from_checkpoint(path, env):
    full_path = os.path.abspath(path)
    print(f"ckpts: {full_path}")
    abstract_mlp = MLP(
        int(env.observation_space.shape[0]),
        int(env.action_space.n),
        [512, 512],
        "relu",
        nnx.Rngs(0),
    )

    graphdef, abstract_state = nnx.split(abstract_mlp)
    checkpointer = ocp.StandardCheckpointer()
    restored_model = checkpointer.restore(full_path, abstract_state)

    q = nnx.merge(graphdef, restored_model)

    def policy(obs):
        return int(jnp.argmax(q([obs])))

    return policy

def get_all_checkpoints(path):
    chkpt_paths = []
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            chkpt_paths.append(full)

    return sorted(chkpt_paths, reverse=True)