from aim import Repo
import matplotlib.pyplot as plt

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

    plt.plot(steps, values, label=f"task {subtask}")

plt.xlabel("Step")
plt.ylabel("Remaining weights (%)")
plt.title("Subnetwork Weights Retained During Pruning")
plt.legend()

plt.xlim(left=0)
plt.show()