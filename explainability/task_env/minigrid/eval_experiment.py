import os
from minigrid.core.constants import COLOR_NAMES
from minigrid_envs import make_ocean_env
from eval_helper import (
    eval_policy,
    get_all_checkpoints,
    get_policy_from_checkpoint,
)

# Used to map colors to integers
# COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}

def eval_all_checkpoints(
    base,
) -> None: 
    all = get_all_checkpoints(base)

    for idx, path in enumerate(all):
        step = int(path.split("_")[-3])
        print(f"step: {step}")


        policy = get_policy_from_checkpoint(path, env)
        # COLOR_NAMES = ["blue"]
        for color in COLOR_NAMES:
            eval_env = make_ocean_env(color = color, render_mode = "human")
            _ = eval_policy(color, eval_env, policy, verbose=True, num_episode=1)

benchmark, grid_size = "oceans", "medium"
seeds = [3, 6, 8, 9]  # successful seeds
# seeds = range(10)

env = make_ocean_env(color = "red", render_mode = "human")

for seed in seeds:
    print(f"seed: {seed}")
    
    base = os.path.expanduser(
        f"~/workspace/XRL/ocean_trained_model/{benchmark}_{grid_size}/uts/seed_{seed}/"
    )
    eval_all_checkpoints(
        base,
    )