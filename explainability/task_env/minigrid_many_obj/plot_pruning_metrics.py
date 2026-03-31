from aim import Repo
import matplotlib.pyplot as plt

file_name = "subnetwork_pruning_weights.pdf"
save_fig = False
font_size = 16
label_pad = 1
title_pad = 25

subnet_colors = {
    "red": "#FF6666",      # moderately light red
    "blue": "#66B2FF",     # moderately light blue
    "purple": "#A566FF",   # moderately light purple
    "grey": "#999999"      # moderately light grey
}

repo = Repo(".")  # path to your Aim repo
run_hash = ["bed5d8d8f2054ac9a45ae783","4622f6a875d74ad9a3f8574d","0e02dd4be3684817b54736af","951c1146502c47fb8df4e8be"]

metrics = repo.query_metrics(
    f"metric.name == 'hard sparsity' and run.hash in {run_hash}"
)

for metric in metrics:
    print("metric from run:", metric.run.name)

    data = list(metric.data)  # your raw structure

    steps = []
    values = []

    for step, (value, epoch, timestamp) in data:
        steps.append(step)
        values.append(value*100)

    experiment = metric.run.experiment
    subtask = experiment.split("_")[1]

    plt.plot(steps, values, color=subnet_colors[subtask], label=f"task {subtask}")

plt.xlabel("step", fontsize=font_size)
plt.ylabel("remaining weights (%)", fontsize=font_size, labelpad=label_pad)
plt.title("Subnetwork Weights Retained During Pruning", fontsize=font_size, pad=title_pad)
plt.legend(loc='lower right', fontsize=font_size)

plt.tick_params(axis='x', labelsize=font_size)
plt.tick_params(axis='y', labelsize=font_size)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.tight_layout()

if save_fig:
    import os
    path = os.path.expanduser(f'~/Pictures/{file_name}')
    plt.savefig(path)
plt.show()