import os
import numpy as np
import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp
from rl_blox.blox.function_approximator.mlp import MLP
# import jax.debug


def eval_policy(color, env, policy, verbose=False, num_episode=1, seed=42):
    task_score_info = {}
    sum_reward = 0.0
    num_task_success = 0.0

    obs, _ = env.reset(seed=seed)
    color_coords = getattr(env.env, f"{color}_coords")
    num_obj_per_color = len(color_coords)
    # print(f"color_coords: {color_coords}, num_obj_per_color: {num_obj_per_color}")
    rng = np.random.default_rng(seed=seed)
    env_seeds = rng.integers(10000, size=num_episode).tolist()
    
    for eps in range(num_episode):
        obs, _ = env.reset(seed=env_seeds[eps])
        terminated = truncated = False

        while not (terminated or truncated):
            action = policy(obs)
            # print(f"action: {action}")
            obs, reward, terminated, truncated, _ = env.step(action)
            # jax.debug.print("obs={obs}", obs=obs)
            sum_reward += reward
            if reward > 0.0:
                num_task_success += 1

    task_score_info.update({
        f"Task {color}": {
            "Avg Return": sum_reward / num_episode,
            "Success Rate": num_task_success / (num_episode*num_obj_per_color)
        }
    })

    if verbose:
        print(task_score_info)

    return task_score_info


def get_policy_from_checkpoint(path, env):
    full_path = os.path.abspath(path)
    print(f"ckpts: {full_path}")
    abstract_mlp = MLP(
        int(env.observation_space.shape[0]),
        int(env.action_space.n),
        [32, 32],
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