import json
from datetime import datetime
from os import walk

from matplotlib import pyplot as plt


def save_model_stats(
    top_n_domains,
    accs,
    maes,
    mses,
    model_name: str,
    acc_baseline=None,
    mae_baseline=None,
    mse_baseline=None,
):
    fig, ax = plt.subplots(2, 2, constrained_layout=True)

    ax[0, 0].set_title("Accuracy")
    ax[0, 0].plot(top_n_domains, accs)
    if acc_baseline:
        ax[0, 0].axhline(y=acc_baseline, color="r", linestyle="-")
    ax[0, 0].set_xlabel("Top domains")
    ax[0, 0].set_ylabel("Accuracy score")

    ax[0, 1].set_title("MAE")
    ax[0, 1].plot(top_n_domains, maes)
    if mae_baseline:
        ax[0, 1].axhline(y=mae_baseline, color="r", linestyle="-")
    ax[0, 1].set_xlabel("Top domains")
    ax[0, 1].set_ylabel("MAE")

    ax[1, 0].set_title("MSE")
    ax[1, 0].plot(top_n_domains, mses)
    if mse_baseline:
        ax[0, 1].axhline(y=mse_baseline, color="r", linestyle="-")
    ax[1, 0].set_xlabel("Top domains")
    ax[1, 0].set_ylabel("MSE")

    ax[1, 1].set_visible(False)

    plt.suptitle(f"Stats for {model_name} on test")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    plt.savefig(
        f"stats_{model_name.lower()}_{timestamp}.png",
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

    for result in files:
        if result == ".DS_Store":
            continue

        with open(f"{results_dir}/{result}") as f:
            data = json.load(f)
            model_name = data["name"]

            domains.append(data["top_n_domains"])
            acc.append(data["classification_report"]["accuracy"])
            mse.append(data["mse"])
            mae.append(data["mae"])

    domains, acc, mse, mae = zip(*sorted(zip(domains, acc, mse, mae)))

    save_model_stats(
        top_n_domains=domains, accs=acc, maes=mae, mses=mse, model_name=model_name
    )


def main():
    plot_results(results_dir="./plot")


if __name__ == "__main__":
    main()
