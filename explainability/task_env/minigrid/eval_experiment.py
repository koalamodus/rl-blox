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
    all_ckpts = get_all_checkpoints(base)
    all_ckpt_results = {}

    for idx, path in enumerate(all_ckpts):
        step = int(path.split("_")[-3])
        print(f"step: {step}")

        ckpt_result = {}

        policy = get_policy_from_checkpoint(path, env)
        for color in COLOR_NAMES:
            eval_env = make_ocean_env(color=color, render_mode=None)
            result = eval_policy(color, eval_env, policy, verbose=False)


            ckpt_result.update(result)
        all_ckpt_results[f"step_{step}"] = ckpt_result
    print_eval_results(all_ckpt_results)


def print_eval_results(all_results):
    if not all_results:
        print("No evaluation results to display.")
        return

    print("\n=== Evaluation Results ===")

    for step in sorted(all_results.keys()):
        print(step)

        for task, result in all_results[step].items():
            print(f" {task}: {result}")
    
    print("\n==========================")


benchmark, grid_size = "oceans", "medium"
seeds = [3, 6, 8, 9]  # successful seeds
# seeds = range(10)

env = make_ocean_env(color = "red", render_mode = "human")

for seed in seeds:
    print(f"Evaluating seed {seed}.")
    
    base = os.path.expanduser(
        f"~/workspace/XRL/ocean_trained_model/{benchmark}_{grid_size}/uts/seed_{seed}/"
    )
    eval_all_checkpoints(
        base,
    )