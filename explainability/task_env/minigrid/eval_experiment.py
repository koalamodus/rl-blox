import os

from minigrid_envs import TASK_COLORS, make_ocean_env
from eval_helper import (
    eval_policy,
    get_all_checkpoints,
    get_policy_from_checkpoint,
)


def eval_all_checkpoints(
    base,
) -> None: 
    all = get_all_checkpoints(base)

    for idx, path in enumerate(all):
        step = int(path.split("_")[-3])
        print(f"step: {step}")


        policy = get_policy_from_checkpoint(path, env)
        
        for color in TASK_COLORS:
            eval_env = make_ocean_env(color = color, render_mode = "human")
            _ = eval_policy(color, eval_env, policy, verbose=True)

color = TASK_COLORS[0]
env = make_ocean_env(color = color, render_mode = "human")


for seed in [0, 4, 9]:
    print(f"seed is {seed}")
    
    base = os.path.expanduser(
        f"~/workspace/XRL/ocean_trained_model/seed_{seed}/"
    )
    eval_all_checkpoints(
        base,
    )