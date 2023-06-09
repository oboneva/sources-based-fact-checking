import json
from datetime import datetime
from typing import List

import pandas as pd

# from mord import LogisticAT
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)

from data_loading_utils import load_datasplits_urls
from metrics_constants import LABELS
from results_utils import save_conf_matrix

# from sklearn.svm import SVC, LinearSVC


NLI_STATS_LABELS = [
    "min_e",
    "min_n",
    "min_c",
    "max_e",
    "max_n",
    "max_c",
    "avg_e",
    "avg_n",
    "avg_c",
]

STANCE_STATS_LABELS = [
    "min_a",
    "min_f",
    "min_o",
    "max_a",
    "max_f",
    "max_o",
    "avg_a",
    "avg_f",
    "avg_o",
]


def save_model_stats(
    mae, mse, report, model_name: str, validate: bool, train_on_val: bool, model_args
):
    test_dataset = "val" if validate else "test"
    train_dataset = "train + val" if train_on_val else "train"

    results = {
        "name": model_name,
        "args": model_args,
        "results_on": test_dataset,
        "trained_on": train_dataset,
        "mae": mae,
        "mse": mse,
        "classification_report": report,
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    with open(f"results_{model_name.lower()}_{timestamp}.json", "w") as outfile:
        json.dump(results, outfile, indent=4)


def load_data_from_urls(articles_dir: str, urls, stats_labels: List[str]):
    infos = []
    for url in urls:
        article_filename = url.split("/")[-2]

        with open(f"{articles_dir}/{article_filename}.json") as f:
            data = json.load(f)

        stats_dict = {
            stats_labels[i]: data["stats"][i] for i in range(len(stats_labels))
        }

        stats_dict["url"] = data["url"]
        stats_dict["label"] = data["label"]
        stats_dict["statsurl"] = data["stats"]

        infos.append(stats_dict)

    return infos


def load_data(urls_path: str, articles_dir: str, stats_labels: List[str]):
    urls_test, urls_val, urls_train = load_datasplits_urls(urls_path=urls_path)

    test_data = load_data_from_urls(
        articles_dir=articles_dir, urls=urls_test, stats_labels=stats_labels
    )
    val_data = load_data_from_urls(
        articles_dir=articles_dir, urls=urls_val, stats_labels=stats_labels
    )
    train_data = load_data_from_urls(
        articles_dir=articles_dir, urls=urls_train, stats_labels=stats_labels
    )

    return (
        test_data,
        val_data,
        train_data,
    )


def encode_label(df, labels_mapper):
    df["label_encoded"] = df["label"].replace(labels_mapper)
    df = df.drop("label", 1)

    return df


def train_test_model(
    model,
    test_data,
    val_data,
    train_data,
    model_args,
    stats_labels: List[str],
    min_max_scale: bool,
    validate=False,
    train_on_val=False,
):
    test_df = pd.DataFrame(test_data)
    val_df = pd.DataFrame(val_data)
    train_df = pd.DataFrame(train_data)

    if train_on_val:
        train_df = pd.concat([val_df, train_df], axis=0)

    # encode labels
    labels_mapper = {LABELS[i]: i + 1 for i in range(len(LABELS))}

    test_df = encode_label(test_df, labels_mapper=labels_mapper)
    if not train_on_val:
        val_df = encode_label(val_df, labels_mapper=labels_mapper)
    train_df = encode_label(train_df, labels_mapper=labels_mapper)

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # separate x and y
    X_train, y_train = train_df[stats_labels], train_df["label_encoded"]
    X_test, y_test = test_df[stats_labels], test_df["label_encoded"]
    if validate:
        X_test, y_test = val_df[stats_labels], val_df["label_encoded"]

    if min_max_scale:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    classification_report_dict = classification_report(
        y_test, y_pred, target_names=LABELS, output_dict=True
    )

    # print("MAF1: %.3f" % classification_report_dict["macro avg"]["f1-score"])
    # print("MAE: %.3f" % mae)
    # print("MSE: %.3f" % mse)

    formater = "{:.3f} {:.3f} {:.3f} {:.4f} {:.4f}"
    formatted_string = formater.format(
        classification_report_dict["accuracy"] * 100,
        classification_report_dict["macro avg"]["f1-score"] * 100,
        classification_report_dict["macro avg"]["recall"] * 100,
        mae,
        mse,
    ).replace(".", ",")

    print(formatted_string)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=[1, 2, 3, 4, 5, 6], display_labels=LABELS
    )

    model_name = type(model).__name__
    save_conf_matrix(disp=disp, model_name=model_name)

    save_model_stats(
        mae=mae,
        mse=mse,
        report=classification_report_dict,
        model_name=model_name,
        validate=validate,
        train_on_val=train_on_val,
        model_args=model_args,
    )


def classification_by_stats(
    urls_path: str,
    articles_dir: str,
    stats_labels: List[str],
    validate=False,
    train_on_val=False,
):
    test_data, val_data, train_data = load_data(
        articles_dir=articles_dir,
        urls_path=urls_path,
        stats_labels=stats_labels,
    )

    lr_model_args = {
        "max_iter": 400,
        "class_weight": "balanced",
        "C": 0.095,
    }
    model = LogisticRegression(**lr_model_args)

    print(lr_model_args)

    train_test_model(
        model=model,
        test_data=test_data,
        val_data=val_data,
        train_data=train_data,
        validate=validate,
        stats_labels=stats_labels,
        train_on_val=train_on_val,
        model_args=lr_model_args,
        min_max_scale=False,
    )


def main():
    classification_by_stats(
        urls_path="./data/urls_split_stratified.json",
        articles_dir="./data/articles_nli",
        stats_labels=NLI_STATS_LABELS,
        validate=False,
        train_on_val=False,
    )

    pass


if __name__ == "__main__":
    main()
