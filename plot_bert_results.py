import json
from datetime import datetime
from os import walk

from matplotlib import pyplot as plt

from metrics_constants import (
    ACC_TEST_DATE,
    ACC_TEST_STRAT,
    AVG_RECALL,
    F1_TEST_STRAT,
    MAE_TEST_DATE,
    MAE_TEST_STRAT,
    MSE_TEST_DATE,
    MSE_TEST_STRAT,
)


def plot_metric(names, values, metric_name, model_name: str, baseline=None):
    _, ax = plt.subplots(1, 1, constrained_layout=True)

    ax.set_title(metric_name)
    ax.barh(names, values)

    if baseline:
        ax.axvline(x=baseline, color="r", linestyle="-")

    plt.suptitle("Stats for test split")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    plt.savefig(
        f"stats_{metric_name.lower()}_{model_name}_{timestamp}.png",
        dpi=300,
    )


def plot_results(results_dir, model_name: str, stratified_split: bool):
    dirs = []
    for (dirpath, dirnames, filenames) in walk(results_dir):
        dirs.extend(dirnames)
        break

    model_names = []
    acc = []
    f1 = []
    mse = []
    mae = []
    recall = []

    for folder in dirs:
        with open(f"{results_dir}/{folder}/test_results.json") as f:
            data = json.load(f)

            folder_name = folder[7:]

            model_names.append(folder_name)
            acc.append(data["test_accuracy"])
            f1.append(data["test_f1"])
            mse.append(data["test_mse"])
            mae.append(data["test_mae"])
            if data["test_recall"]:
                recall.append(data["test_recall"])

    acc, model_names1 = zip(*sorted(zip(acc, model_names)))
    acc_b = ACC_TEST_STRAT if stratified_split else ACC_TEST_DATE
    plot_metric(model_names1, acc, "Accuracy", model_name, acc_b)

    f1, model_names2 = zip(*sorted(zip(f1, model_names)))
    f1_b = F1_TEST_STRAT if stratified_split else None
    plot_metric(model_names2, f1, "Macro avg F1", model_name, f1_b)

    mae, model_names3 = zip(*sorted(zip(mae, model_names), reverse=True))
    mae_b = MAE_TEST_STRAT if stratified_split else MAE_TEST_DATE
    plot_metric(model_names3, mae, "MAE", model_name, mae_b)

    mse, model_names4 = zip(*sorted(zip(mse, model_names), reverse=True))
    mse_b = MSE_TEST_STRAT if stratified_split else MSE_TEST_DATE
    plot_metric(model_names4, mse, "MSE", model_name, mse_b)

    if len(recall) > 0:
        recall, model_names5 = zip(*sorted(zip(recall, model_names)))
        plot_metric(model_names5, recall, "Macro avg recall", model_name, AVG_RECALL)


def main():
    plot_results(
        results_dir="./results/outputs", model_name="bert", stratified_split=True
    )


if __name__ == "__main__":
    main()
