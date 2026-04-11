import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def plot_heatmaps(
    full_scores,
    subnet_scores,
    relative_performance,
    subtasks,
    metric="Success Rate",
    figsize=(10, 6),
    cmap="viridis",
    annot=True,
):
    tasks = list(full_scores.keys())
    x_labels = [f"task {t.split('Task ')[-1]}" for t in tasks]

    # ---------------------------
    # Absolute performance matrix
    # ---------------------------
    abs_df = pd.DataFrame(index=x_labels)

    # full network
    abs_df["full_network"] = [
        full_scores[t][metric] for t in tasks
    ]

    # subnetworks
    for subtask in subtasks:
        abs_df[f"subnet_{subtask}"] = [
            subnet_scores[subtask][t][metric] for t in tasks
        ]

    # ---------------------------
    # Relative performance matrix
    # ---------------------------
    rel_df = pd.DataFrame(index=x_labels)

    for subtask in subtasks:
        rel_df[f"subnet_{subtask}"] = [
            relative_performance[subtask][t][metric] for t in tasks
        ]

    # ---------------------------
    # Plot
    # ---------------------------
    fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 2))

    # --- Absolute heatmap ---
    sns.heatmap(
        abs_df,
        ax=axes[0],
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
    )
    axes[0].set_title("Absolute Performance")

    # --- Relative heatmap ---
    sns.heatmap(
        rel_df,
        ax=axes[1],
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
    )
    axes[1].set_title("Relative Performance")

    plt.tight_layout()
    plt.show()



# ---------------------------
# Set env
# ---------------------------
subtasks = ["red", "blue", "purple", "grey"]
# save_plot = True
# metric_to_plot = "Success Rate"
metric_to_plot = "Avg Return"
# ---------------------------
# Load evaluation result
# ---------------------------
import os
import pickle

experiment = "reefshield_random_medium"
file_name = "minigrid_evaluation_results.pkl"

eval_result_path = os.path.expanduser(
        f"~/workspace/XRL/ocean_subnet/{experiment}/{file_name}"
    )
# eval_result_path = "evaluation_results.pkl"
with open(eval_result_path, "rb") as f:
    eval_results = pickle.load(f)

# ---------------------------
# Build tables
# ---------------------------
full_scores = eval_results["full_scores"]
subnet_scores = eval_results["subnet_scores"]
relative_performance = eval_results["relative_performance"]

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

# ---------------------------
# Heatmaps
# ---------------------------
plot_heatmaps(
    full_scores=full_scores,
    subnet_scores=subnet_scores,
    relative_performance=relative_performance,
    subtasks=subtasks,
    metric=metric_to_plot,
)
