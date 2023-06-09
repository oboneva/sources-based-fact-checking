import json
import statistics
from datetime import datetime
from math import sqrt
from os import walk

from matplotlib import pyplot as plt

from metrics_constants import (
    ACC_TEST_DATE,
    ACC_TEST_STRAT,
    ACC_VAL_DATE,
    ACC_VAL_STRAT,
    F1_TEST_STRAT,
    F1_VAL_STRAT,
    MAE_TEST_DATE,
    MAE_TEST_STRAT,
    MAE_VAL_DATE,
    MAE_VAL_STRAT,
    MSE_TEST_DATE,
    MSE_TEST_STRAT,
    MSE_VAL_DATE,
    MSE_VAL_STRAT,
)


def save_model_stats(
    top_n_domains,
    accs,
    maes,
    mses,
    f1s,
    model_name: str,
    results_on_val: bool,
    stratified: bool,
    add_baseline: bool,
):
    fig, ax = plt.subplots(2, 2, constrained_layout=True)

    mae_x, mae_y = 0, 0
    mse_x, mse_y = 0, 1
    acc_x, acc_y = 1, 1
    f1_x, f1_y = 1, 0

    ax[acc_x, acc_y].set_title("Accuracy")
    ax[acc_x, acc_y].plot(top_n_domains, accs)

    if add_baseline:
        acc_baseline = (
            (ACC_VAL_STRAT if results_on_val else ACC_TEST_STRAT)
            if stratified
            else (ACC_VAL_DATE if results_on_val else ACC_TEST_DATE)
        )
        ax[acc_x, acc_y].axhline(y=acc_baseline, color="r", linestyle="-")
    ax[acc_x, acc_y].set_xlabel("Top n domains")
    ax[acc_x, acc_y].set_ylabel("Accuracy score")
    ax[acc_x, acc_y].grid()

    ax[mae_x, mae_y].set_title("MAE")
    ax[mae_x, mae_y].plot(top_n_domains, maes)
    if add_baseline:
        mae_baseline = (
            (MAE_VAL_STRAT if results_on_val else MAE_TEST_STRAT)
            if stratified
            else (MAE_VAL_DATE if results_on_val else MAE_TEST_DATE)
        )
        ax[mae_x, mae_y].axhline(y=mae_baseline, color="r", linestyle="-")
    ax[mae_x, mae_y].set_xlabel("Top n domains")
    ax[mae_x, mae_y].set_ylabel("MAE")
    ax[mae_x, mae_y].grid()

    ax[mse_x, mse_y].set_title("MSE")
    ax[mse_x, mse_y].plot(top_n_domains, mses)
    if add_baseline:
        mse_baseline = (
            (MSE_VAL_STRAT if results_on_val else MSE_TEST_STRAT)
            if stratified
            else (MSE_VAL_DATE if results_on_val else MSE_TEST_DATE)
        )
        ax[mse_x, mse_y].axhline(y=mse_baseline, color="r", linestyle="-")
    ax[mse_x, mse_y].set_xlabel("Top n domains")
    ax[mse_x, mse_y].set_ylabel("MSE")
    ax[mse_x, mse_y].grid()

    # ax[1, 1].set_visible(False)
    ax[f1_x, f1_y].set_title("F1 score")
    ax[f1_x, f1_y].plot(top_n_domains, f1s)
    if add_baseline:
        f1_baseline = (
            (F1_VAL_STRAT if results_on_val else F1_TEST_STRAT)
            if stratified
            else (MSE_VAL_DATE if results_on_val else MSE_TEST_DATE)
        )
        ax[f1_x, f1_y].axhline(y=f1_baseline, color="r", linestyle="-")
    ax[f1_x, f1_y].set_xlabel("Top n domains")
    ax[f1_x, f1_y].set_ylabel("F1 score")
    ax[f1_x, f1_y].grid()

    dataset = "val" if results_on_val else "test"
    split = "stratified" if stratified else "date"

    plt.suptitle(f"Stats for {model_name} on {dataset} {split} split")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    plt.savefig(
        f"stats_{dataset}_{split}_{model_name.lower()}_{timestamp}.png",
        dpi=300,
    )


def save_conf_matrix(disp, model_name: str, top_n_domains=None):
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)

    plot_title = (
        f"{model_name}, Top {top_n_domains} domains" if top_n_domains else model_name
    )

    plt.title(plot_title)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    filename = (
        f"conf_matrix_{model_name.lower()}_top_{top_n_domains}_{timestamp}.png"
        if top_n_domains
        else f"conf_matrix_{model_name.lower()}_{timestamp}.png"
    )

    plt.savefig(filename, pad_inches=5, dpi=300)


def compute_mean_std(values, clip):
    std = statistics.stdev(values[clip:])
    m = statistics.mean(values[clip:])
    return m, std


def plot_results(results_dir):
    files = []
    for (dirpath, dirnames, filenames) in walk(results_dir):
        files.extend(filenames)
        break

    model_name = ""
    domains = []
    acc = []
    mse = []
    mae = []
    f1s = []
    rmse = []
    recall = []

    for result in files:
        if result == ".DS_Store":
            continue

        with open(f"{results_dir}/{result}") as f:
            data = json.load(f)
            model_name = data["name"]

            domains.append(data["top_n_domains"])
            acc.append(data["classification_report"]["accuracy"] * 100)
            mse.append(data["mse"])
            mae.append(data["mae"])
            f1s.append(data["classification_report"]["macro avg"]["f1-score"] * 100)
            rmse.append(sqrt(data["mse"]))
            recall.append(data["classification_report"]["macro avg"]["recall"] * 100)

    domains, acc, mse, mae, f1s, rmse, recall = zip(
        *sorted(zip(domains, acc, mse, mae, f1s, rmse, recall))
    )

    print("min mae", min(mae))
    print("min mse", min(mse))
    print("max acc", max(acc))
    print("max f1s", max(f1s))
    print("max recall", max(recall))

    print("\n")

    m, std = compute_mean_std(mae, 12)
    print("mae: {:.3f} {:.3f}".format(m, std))

    m, std = compute_mean_std(mse, 12)
    print("mse: {:.3f} {:.3f}".format(m, std))

    m, std = compute_mean_std(rmse, 12)
    print("rmse: {:.3f} {:.3f}".format(m, std))

    m, std = compute_mean_std(acc, 12)
    print("acc: {:.2f} {:.2f}".format(m, std))

    m, std = compute_mean_std(f1s, 12)
    print("f1: {:.2f} {:.2f}".format(m, std))

    m, std = compute_mean_std(recall, 12)
    print("recall: {:.2f} {:.2f}".format(m, std))

    for i in zip(domains, acc, f1s, recall, mae, mse, rmse):
        d, ac, f1, r, ma, ms, rms = i

        formater = "{:.2f} {:.2f} {:.2f} {:.3f} {:.3f} {:.3f}"
        formatted_string = formater.format(ac, f1, r, ma, ms, rms)

        print(d, formatted_string)
    # print(i)

    save_model_stats(
        top_n_domains=domains,
        f1s=f1s,
        accs=acc,
        maes=mae,
        mses=mse,
        model_name=model_name,
        results_on_val=False,
        stratified=True,
        add_baseline=False,
    )


def main():
    plot_results(results_dir="domains/svr/results")


if __name__ == "__main__":
    main()
