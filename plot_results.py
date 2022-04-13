import json
from datetime import datetime
from os import walk

from matplotlib import pyplot as plt


def plot_metric(names, values, metric_name, baseline=None):
    _, ax = plt.subplots(1, 1, constrained_layout=True)

    ax.set_title(metric_name)
    ax.barh(names, values)

    if baseline:
        ax.axvline(x=baseline, color="r", linestyle="-")

    plt.suptitle("Stats for test split")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    plt.savefig(
        f"stats_{metric_name.lower()}_bert_{timestamp}.png",
        dpi=300,
    )


def plot_results(results_dir):
    dirs = []
    for (dirpath, dirnames, filenames) in walk(results_dir):
        dirs.extend(dirnames)
        break

    model_names = []
    acc = []
    f1 = []
    mse = []
    mae = []

    for folder in dirs:
        with open(f"{results_dir}/{folder}/test_results.json") as f:
            data = json.load(f)

            folder_name = folder[7:]

            model_names.append(folder_name)
            acc.append(data["test_accuracy"])
            f1.append(data["test_f1"])
            mse.append(data["test_mse"])
            mae.append(data["test_mae"])

    acc, model_names1 = zip(*sorted(zip(acc, model_names)))
    plot_metric(model_names1, acc, "Accuracy", 0.481829)

    f1, model_names2 = zip(*sorted(zip(f1, model_names)))
    plot_metric(model_names2, f1, "Macro avg F1")

    mae, model_names3 = zip(*sorted(zip(mae, model_names), reverse=True))
    plot_metric(model_names3, mae, "MAE", 0.8611311672683514)

    mse, model_names4 = zip(*sorted(zip(mse, model_names), reverse=True))
    plot_metric(model_names4, mse, "MSE", 1.9068592057761733)


def main():
    plot_results(results_dir="./results/outputs")


if __name__ == "__main__":
    main()
