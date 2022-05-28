import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
)

from data_loading_utils import load_datasplits_urls
from labels_mapping_utils import create_label2id_mapper


def load_data_from_urls(articles_dir: str, labels_mapper, urls):
    targets = []
    for url in urls:
        article_filename = url.split("/")[-2]

        with open(f"{articles_dir}/{article_filename}.json") as f:
            data = json.load(f)

        targets.append(labels_mapper[data["label"]])

    return targets


def compute_mae_mse_baselines(articles_dir: str, urls_path: str, labels_mapper):
    urls_test, urls_val, urls_train = load_datasplits_urls(urls_path=urls_path)

    y_test = load_data_from_urls(
        articles_dir=articles_dir, labels_mapper=labels_mapper, urls=urls_test
    )
    y_val = load_data_from_urls(
        articles_dir=articles_dir, labels_mapper=labels_mapper, urls=urls_val
    )
    y_train = load_data_from_urls(
        articles_dir=articles_dir, labels_mapper=labels_mapper, urls=urls_train
    )

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

    for i in range(len(set(labels_mapper.values()))):
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

    # print("Val MAE baseline: ", min(val_maes))
    # print("Val MSE baseline: ", min(val_mses))

    # print("Train MAE baseline: ", min(train_maes))
    # print("Train MSE baseline: ", min(train_mses))


def compute_acc_recall_f1_baseline(
    articles_dir: str, urls_path: str, labels_mapper, label: str
):
    urls_test, urls_val, urls_train = load_datasplits_urls(urls_path=urls_path)

    def compute_acc_recall_f1_baseline_for_data_split(urls, split_name):
        y_split = load_data_from_urls(
            articles_dir=articles_dir, labels_mapper=labels_mapper, urls=urls
        )

        y_pred_single_class = np.full((len(y_split), 1), labels_mapper[label])

        f1_split = f1_score(y_split, y_pred_single_class, average="macro")
        accuracy = accuracy_score(y_true=y_split, y_pred=y_pred_single_class)
        recall = recall_score(
            y_true=y_split, y_pred=y_pred_single_class, average="macro"
        )

        print(f"{split_name} Macro Avg F1: ", f1_split)
        print(f"{split_name} Macro Avg Recall: ", recall)
        print(f"{split_name} Accuracy: ", accuracy)

    compute_acc_recall_f1_baseline_for_data_split(urls_test, "Test")
    # compute_acc_recall_f1_baseline_for_data_split(urls_val, "Val")
    # compute_acc_recall_f1_baseline_for_data_split(urls_train, "Train")


def main():
    labels_mapper = create_label2id_mapper(num_classes=4)

    print(labels_mapper)

    compute_acc_recall_f1_baseline(
        articles_dir="./data/articles_parsed_clean_date",
        urls_path="./data/urls_split_stratified.json",
        labels_mapper=labels_mapper,
        label="half-true",
    )

    compute_mae_mse_baselines(
        articles_dir="./data/articles_parsed_clean_date",
        urls_path="./data/urls_split_stratified.json",
        labels_mapper=labels_mapper,
    )


if __name__ == "__main__":
    main()
