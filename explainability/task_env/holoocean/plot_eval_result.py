from holoocean_envs import TASK_NAMES

# ["red", "blue", "green", "black"]
subtasks = TASK_NAMES
save_fig = True

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

full_scores = eval_results["full_scores"]
subnet_scores = eval_results["subnet_scores"]
relative_performance = eval_results["performance_drop"]


# ---------------------------
# Plot subnetwork performance
# ---------------------------

import matplotlib.pyplot as plt
import numpy as np

subnet_colors = {
    "red": "#FF6666",      # moderately light red
    "blue": "#66B2FF",     # moderately light blue
    "green": "#66CC99",    # moderately light green
    "black": "#999999"     # moderately light grey
}
full_color="#444444"
full_network_label="full network"

metric_to_plot = "Success Rate"
# metric_to_plot = "Avg Return"
font_size=28
label_pad=15
y_label_coords=[-0.12, 0.5]


def plot_full_subnet_with_drop(
    full_scores,
    subnet_scores,
    relative_performance,
    subtasks,
    subnet_colors,
    metric,
    full_network_label=full_network_label,
    full_color=full_color,
):
    """
    Top: Full + subnet performance
    Bottom: Performance drop (aligned, no bar for full network)

    Ensures vertical alignment of bars across subplots.
    """

    y_label_abs = {
        "Avg Return": "average return",
        "Success Rate": "object found rate",
    }

    y_label_rel = {
        "Avg Return": "relative performance",
        "Success Rate": "relative performance",
    }

    file_name = {
        "Avg Return": "holoocean_avg_return",
        "Success Rate": "holoocean_success_rate",
    }

    tasks = list(full_scores.keys())
    x_tick_label = [f"task {t.split('Task ')[-1]}" for t in tasks]

    n_tasks = len(tasks)
    n_subnets = len(subtasks)

    total_bars = n_subnets + 1  # full + subnets
    bar_width = 0.8 / total_bars
    x = np.arange(n_tasks)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ---------- TOP: absolute performance ----------
    ax = axes[0]

    # Full network (leftmost position)
    full_values = [full_scores[task][metric] for task in tasks]

    ax.bar(
        x,
        full_values,
        width=bar_width,
        color=full_color,
        label=full_network_label,
    )

    # Subnetworks
    for i, subtask in enumerate(subtasks):
        values = [subnet_scores[subtask][task][metric] for task in tasks]
        ax.bar(
            x + (i + 1) * bar_width,
            values,
            width=bar_width,
            color=subnet_colors[subtask],
            label=f"subnetwork {subtask}",
        )

    ax.set_ylabel(y_label_abs[metric], fontsize=font_size, labelpad=label_pad)
    ax.yaxis.set_label_coords(y_label_coords[0], y_label_coords[1])
    # ax.set_title("Absolute Performance", fontsize=font_size)
    if metric == "Success Rate":
        ax.set_ylim(0., 1.0)
    ax.grid(axis="y")
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.6),  # move above plot
        ncol=2,
        # handlelength=1.5,     # length of color bar
        handletextpad=0.8,      # space between color bar and label text
        labelspacing=0.3,       # vertical spacing between legend entries
        columnspacing=1.0,      # space between legend columns
        fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)  # x-axis numbers
    ax.tick_params(axis='y', labelsize=font_size)  # y-axis numbers

    # ---------- BOTTOM: performance drop ----------
    ax = axes[1]

    # IMPORTANT: skip full network slot → leave x empty
    for i, subtask in enumerate(subtasks):
        values = [
            relative_performance[subtask][task][metric]
            for task in tasks
        ]
        ax.bar(
            x + (i + 1) * bar_width,  # SAME POSITION as above
            values,
            width=bar_width,
            color=subnet_colors[subtask],
        )

    ax.set_ylabel(y_label_rel[metric], fontsize=font_size, labelpad=label_pad)
    # ax.set_title("Relative Performance", fontsize=font_size)
    if metric == "Success Rate":
        ax.set_ylim(0., 1.0)
    ax.grid(axis="y")
    ax.yaxis.set_label_coords(y_label_coords[0], y_label_coords[1])
    ax.tick_params(axis='x', labelsize=font_size)  # x-axis numbers
    ax.tick_params(axis='y', labelsize=font_size)  # y-axis numbers

    # Shared X axis (centered across ALL slots including empty full slot)
    axes[1].set_xticks(x + bar_width * (total_bars - 1) / 2)
    axes[1].set_xticklabels(x_tick_label, fontsize=font_size)
    # axes[1].set_xlabel("Task", fontsize=font_size)

    # fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(hspace=0.2)

    if save_fig:
        import os
        path = os.path.expanduser(f'~/Pictures/{file_name[metric]}.pdf')
        plt.savefig(path)

    plt.show()

plot_full_subnet_with_drop(
    full_scores=full_scores,
    subnet_scores=subnet_scores,
    relative_performance=relative_performance,
    subtasks=subtasks,
    subnet_colors=subnet_colors,
    full_color=full_color,
    metric=metric_to_plot,
)