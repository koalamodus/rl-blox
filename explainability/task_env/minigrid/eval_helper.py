import os

import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp
from rl_blox.blox.function_approximator.mlp import MLP


def eval_policy(color, env, policy, verbose=False):
    task_scores = {}
    ep_reward = 0.0
    obs, _ = env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        action = policy(obs)
        # print(f"action: {action}")
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward

    task_scores.update({f"Task {color}": [ep_reward]})

    if verbose:
        print(task_scores)

    return task_scores


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

    return sorted(chkpt_paths)