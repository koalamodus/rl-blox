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

tasks = [f"Task {t}" for t in subtasks]
x_tick_label = [f"task {t}" for t in subtasks]
n_tasks = len(tasks)
n_subnets = len(subtasks)

# Bar width and positions
bar_width = 0.1
x = np.arange(n_tasks)  # task positions

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, metric in zip(axes, metric_to_plot):
    for i, subtask in enumerate(subtasks):
        drops = [performance_drop[subtask][task][metric] for task in tasks]
        ax.bar(
            x + i*bar_width,
            drops,
            width=bar_width,
            color=subnet_colors[subtask],
            label=f"subnetwork {subtask}"
        )
    ax.set_xticks(x + bar_width*(n_subnets-1)/2)
    ax.set_xticklabels(x_tick_label)
    ax.set_xlabel("Task")
    ax.set_title(metric)
    ax.grid(axis='y')

axes[0].set_ylabel("Performance Drop")
axes[0].legend()

fig.suptitle("Performance Drop of Subnetworks Relative to the Full Network", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()