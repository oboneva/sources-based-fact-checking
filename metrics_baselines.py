import json

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

from data_loading_utils import load_datasplits_urls
from metrics_constants import LABELS


def load_data_from_urls(articles_dir: str, urls):
    labels_mapper = {LABELS[i]: i for i in range(len(LABELS))}

    targets = []
    for url in urls:
        article_filename = url.split("/")[-2]

        with open(f"{articles_dir}/{article_filename}.json") as f:
            data = json.load(f)

        targets.append(labels_mapper[data["label"]])

    return targets


def compute_mae_mse_baselines(articles_dir: str, urls_path: str):
    urls_test, urls_val, urls_train = load_datasplits_urls(urls_path=urls_path)

    y_test = load_data_from_urls(articles_dir=articles_dir, urls=urls_test)
    y_val = load_data_from_urls(articles_dir=articles_dir, urls=urls_val)
    y_train = load_data_from_urls(articles_dir=articles_dir, urls=urls_train)

    def compute_mae_mse_baselines_for_data_split(y_split, single_y, split_name):
        y_pred_single_class = np.full((len(y_split), 1), single_y)

        mae = mean_absolute_error(y_split, y_pred_single_class)
        mse = mean_squared_error(y_split, y_pred_single_class)

        print(f"{split_name} MAE: ", mae)
        print(f"{split_name} MSE: ", mse)

        return mae, mse

    test_maes = []
    val_maes = []
    train_maes = []

    test_mses = []
    val_mses = []
    train_mses = []

    for i in range(6):
        print("\nClass: ", i)

        mae_test, mse_test = compute_mae_mse_baselines_for_data_split(y_test, i, "Test")
        mae_val, mse_val = compute_mae_mse_baselines_for_data_split(y_val, i, "Val")
        mae_train, mse_train = compute_mae_mse_baselines_for_data_split(
            y_train, i, "Train"
        )

        test_maes.append(mae_test)
        val_maes.append(mae_val)
        train_maes.append(mae_train)

        test_mses.append(mse_test)
        val_mses.append(mse_val)
        train_mses.append(mse_train)

    print("\nBaselines")

    print("Test MAE baseline: ", min(test_maes))
    print("Test MSE baseline: ", min(test_mses))

    print("Val MAE baseline: ", min(val_maes))
    print("Val MSE baseline: ", min(val_mses))

    print("Train MAE baseline: ", min(train_maes))
    print("Train MSE baseline: ", min(train_mses))


def compute_f1_baseline(articles_dir: str, urls_path: str):
    urls_test, urls_val, urls_train = load_datasplits_urls(urls_path=urls_path)

    labels_mapper = {LABELS[i]: i for i in range(len(LABELS))}

    def compute_f1_baseline_for_data_split(urls, split_name):
        y_split = load_data_from_urls(articles_dir=articles_dir, urls=urls)

        y_pred_single_class = np.full((len(y_split), 1), labels_mapper["false"])

        f1_split = f1_score(y_split, y_pred_single_class, average="macro")

        print(f"{split_name} Macro Avg F1: ", f1_split)

    compute_f1_baseline_for_data_split(urls_test, "Test")
    compute_f1_baseline_for_data_split(urls_val, "Val")
    compute_f1_baseline_for_data_split(urls_train, "Train")


def main():
    compute_f1_baseline(
        articles_dir="./data/articles_parsed",
        urls_path="./data/urls_split_stratified.json",
    )


if __name__ == "__main__":
    main()
