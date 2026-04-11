import pandas as pd
import numpy as np

def build_performance_tables(
    full_scores,
    subnet_scores,
    relative_performance,
    subtasks,
    metric="Success Rate",
):
    """
    Create tables instead of bar plots:
    1) Absolute performance
    2) Relative performance (drop / ratio)
    """

    tasks = list(full_scores.keys())

    # ---------------------------
    # Absolute performance table
    # ---------------------------
    abs_data = {"task": []}

    # full network column
    abs_data["full_network"] = []

    # subnet columns
    for subtask in subtasks:
        abs_data[f"subnet_{subtask}"] = []

    for task in tasks:
        abs_data["task"].append(task)

        # full network
        abs_data["full_network"].append(full_scores[task][metric])

        # subnets
        for subtask in subtasks:
            abs_data[f"subnet_{subtask}"].append(
                subnet_scores[subtask][task][metric]
            )

    abs_df = pd.DataFrame(abs_data)

    # ---------------------------
    # Relative performance table
    # ---------------------------
    rel_data = {"task": []}

    for subtask in subtasks:
        rel_data[f"subnet_{subtask}"] = []

    for task in tasks:
        rel_data["task"].append(task)

        for subtask in subtasks:
            rel_data[f"subnet_{subtask}"].append(
                relative_performance[subtask][task][metric]
            )

    rel_df = pd.DataFrame(rel_data)

    return abs_df, rel_df

# ---------------------------
# Set env
# ---------------------------
from holoocean_envs import TASK_NAMES

# ["red", "blue", "green", "black"]
subtasks = TASK_NAMES
# save_plot = True
# metric_to_plot = "Success Rate"
metric_to_plot = "Avg Return"
# ---------------------------
# Load evaluation result
# ---------------------------
import os
import pickle

experiment = "holoocean"
file_name = "holoocean_evaluation_results.pkl"

eval_result_path = os.path.expanduser(
        f"~/workspace/XRL/ocean_subnet/{experiment}/{file_name}"
    )
with open(eval_result_path, "rb") as f:
    eval_results = pickle.load(f)

# ---------------------------
# Build tables
# ---------------------------
full_scores = eval_results["full_scores"]
subnet_scores = eval_results["subnet_scores"]
relative_performance = eval_results["performance_drop"]

abs_df, rel_df = build_performance_tables(
    full_scores,
    subnet_scores,
    relative_performance,
    subtasks,
    metric=metric_to_plot,
)

# ---------------------------
# Display tables
# ---------------------------
print("\n=== Absolute Performance ===")
print(abs_df.to_string(index=False))

print("\n=== Relative Performance ===")
print(rel_df.to_string(index=False))