import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def build_performance_tables(
    full_scores,
    subnet_scores,
    relative_performance,
    subtasks,
    metric="Avg Return",
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
    x_labels = [f"{t.split('Task ')[-1]}" for t in tasks]

    # ---------------------------
    # Absolute performance matrix
    # ---------------------------
    abs_df = pd.DataFrame(index=x_labels)

    # full network
    abs_df["full network"] = [
        full_scores[t][metric] for t in tasks
    ]

    # subnetworks
    for subtask in subtasks:
        abs_df[f"subnet {subtask}"] = [
            subnet_scores[subtask][t][metric] for t in tasks
        ]

    # ---------------------------
    # Relative performance matrix
    # ---------------------------
    rel_df = pd.DataFrame(index=x_labels)

    for subtask in subtasks:
        rel_df[f"{subtask}"] = [
            relative_performance[subtask][t][metric] for t in tasks
        ]

    # ---------------------------
    # Plot
    # ---------------------------
    line_width = 0.2
    font_size = 12
    x_label = ["Networks", "Subnetworks"]
    y_label = "Tasks"

    fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 2))
    axes[0].set_aspect("auto")
    axes[1].set_aspect("auto")
    axes[0].tick_params(labelsize=8)
    axes[1].tick_params(labelsize=8)

    heatmap_kws = dict(
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        linewidths=0.2,
        linecolor="white",
        cbar=True,
        annot_kws={"size": font_size},
    )

    
    # --- Absolute heatmap ---
    sns.heatmap(abs_df, ax=axes[0], **heatmap_kws)
    axes[0].set_title(f"Normalized {metric}", fontsize=font_size)
    axes[0].tick_params(axis="both", labelsize=font_size)
    axes[0].set_xlabel(x_label[0], fontsize=font_size)
    axes[0].set_ylabel(y_label, fontsize=font_size)


    # --- Relative heatmap ---
    sns.heatmap(rel_df, ax=axes[1], **heatmap_kws)
    axes[1].set_title(f"Relative {metric}", fontsize=font_size)
    axes[1].tick_params(axis="both", labelsize=font_size)
    axes[1].set_xlabel(x_label[1], fontsize=font_size)
    axes[1].set_ylabel(y_label, fontsize=font_size)

    # ---------------------------
    # Colorbar font size fix
    # ---------------------------
    for ax in axes:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=font_size)

    plt.tight_layout()

    if save_fig:
        remap_metric = {
            "Avg Return": "avg_return",
            "Success Rate": "success_rate",
        }
        metric = remap_metric[metric]
        path = os.path.expanduser(f'~/Pictures/{experiment}_{metric}_heaptmap.pdf')
        plt.savefig(path)
    
    plt.show()

def normalize_metric_dict(data, metric, vmin, vmax):
    """
    Works for:
    - full_scores: {task: {metric: value}}
    - subnet_scores[subtask]: {task: {metric: value}}
    """

    def normalize(x):
        if vmax == vmin:
            raise ValueError(f"Cannot normalize when vmax == vmin == {vmin}")
        return (x - vmin) / (vmax - vmin)

    normalized = {}

    for key, metrics in data.items():
        normalized[key] = metrics.copy()
        normalized[key][metric] = normalize(metrics[metric])

    return normalized

def normalize_full_scores(full_scores, metric, vmin, vmax):
    return normalize_metric_dict(full_scores, metric, vmin, vmax)

def normalize_subnet_scores(subnet_scores, subtasks, metric, vmin, vmax):
    normalized = {}

    for subtask in subtasks:
        normalized[subtask] = normalize_metric_dict(
            subnet_scores[subtask],
            metric,
            vmin,
            vmax
        )

    return normalized

def compute_relative_performance(subtasks, subnet_scores, full_scores):
    relative_performance = {}

    for subtask in subtasks:
        scores = subnet_scores[subtask]

        rel_perform = {
            f"Task {t}": {
                metric: scores[f"Task {t}"][metric] / full_scores[f"Task {t}"][metric]
                for metric in full_scores[f"Task {t}"]
            }
            for t in subtasks
        }

        relative_performance[subtask] = rel_perform

    return relative_performance

# ---------------------------
# Set env
# ---------------------------
from holoocean_envs import TASK_NAMES

# ["red", "blue", "green", "black"]
subtasks = TASK_NAMES
save_fig = True
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

# Normalize Avg Return
metric = metric_to_plot
if metric_to_plot == "Avg Return":
    vmin = -1000
    vmax = 1000
full_scores = normalize_metric_dict(full_scores, metric, vmin, vmax)
subnet_scores = {
    s: normalize_metric_dict(subnet_scores[s], metric, vmin, vmax)
    for s in subtasks
}
# Recompute relative avg return

relative_performance = compute_relative_performance(subtasks, subnet_scores, full_scores)

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
