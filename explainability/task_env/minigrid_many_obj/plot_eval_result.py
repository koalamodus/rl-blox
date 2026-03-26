subtasks = ["red", "blue", "purple", "grey"]


# ---------------------------
# Load evaluation result
# ---------------------------
import pickle

with open("evaluation_results.pkl", "rb") as f:
    eval_results = pickle.load(f)

full_scores = eval_results["full_scores"]
subnet_scores = eval_results["subnet_scores"]
performance_drop = eval_results["performance_drop"]


# ---------------------------
# Plot subnetwork performance
# ---------------------------

import matplotlib.pyplot as plt
import numpy as np

subnet_colors = {
    "red": "#FF6666",      # moderately light red
    "blue": "#66B2FF",     # moderately light blue
    "purple": "#A566FF",   # moderately light purple
    "grey": "#999999"      # moderately light grey
}

metric_to_plot = ["Avg Return", "Success Rate"]

import matplotlib.pyplot as plt
import numpy as np

def plot_subnetwork_metrics(data_dict, title, subtasks, subnet_colors, metrics=["Avg Return", "Success Rate"]):
    """
    Plots subnetwork metrics in a 2-row figure (Avg Return on top, Success Rate below).

    Args:
        data_dict (dict): nested dictionary of shape data_dict[subtask][task][metric]
        title (str): suptitle for the figure
        subtasks (list of str): list of subnetwork names
        subnet_colors (dict): mapping of subnetwork name to color
        metrics (list of str): list of metrics to plot (default ["Avg Return", "Success Rate"])
    """
    tasks = [f"Task {t}" for t in subtasks]
    x_tick_label = [f"task {t}" for t in subtasks]
    y_label= {
        "Avg Return": "average return",
        "Success Rate": "success rate"
    }
    n_tasks = len(tasks)
    n_subnets = len(subtasks)
    bar_width = 0.15
    x = np.arange(n_tasks)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    for ax, metric in zip(axes, metrics):
        for i, subtask in enumerate(subtasks):
            values = [data_dict[subtask][task][metric] for task in tasks]
            ax.bar(
                x + i*bar_width,
                values,
                width=bar_width,
                color=subnet_colors[subtask],
                label=f"subnetwork {subtask}"
            )
        ax.set_ylabel(y_label[metric])
        ax.grid(axis='y')
        # ax.set_title(metric)
        ax.set_xticks(x + bar_width*(n_subnets-1)/2)
        ax.set_xticklabels(x_tick_label)

    # # X-axis labels only on the bottom subplot
    # axes[1].set_xticks(x + bar_width*(n_subnets-1)/2)
    # axes[1].set_xticklabels(x_tick_label)
    # axes[1].set_xlabel("Task")

    # Legend only once (top subplot)
    axes[0].legend()
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


plot_subnetwork_metrics(
    data_dict=subnet_scores,
    title="Performance of Subnetworks Across Tasks",
    subtasks=subtasks,
    subnet_colors=subnet_colors,
    metrics=metric_to_plot
)

plot_subnetwork_metrics(
    data_dict=performance_drop,
    title="Performance Drop of Subnetworks Relative to the Full Network",
    subtasks=subtasks,
    subnet_colors=subnet_colors,
    metrics=metric_to_plot
)