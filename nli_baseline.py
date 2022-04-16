import json
from dataclasses import dataclass
from datetime import datetime
from os import walk
from statistics import mean

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.svm import SVC, LinearSVC

from data_loading_utils import load_datasplits_urls
from metrics_constants import LABELS
from nli_model import NLIModel
from results_utils import save_conf_matrix

STATS_LABELS = [
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


@dataclass
class NLISource:
    text: str
    entailment: float
    neutral: float
    contradiction: float


def get_nli_source(source: str, probs):
    return vars(
        NLISource(
            text=source,
            entailment=probs["entailment"],
            neutral=probs["neutral"],
            contradiction=probs["contradiction"],
        )
    )


def save_nli_probs(articles_dir: str, new_articles_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    new_files = []
    for (dirpath, dirnames, filenames) in walk(new_articles_dir):
        new_files.extend(filenames)
        break

    model = NLIModel("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", device)

    for article in files:
        if article == ".DS_Store" or article in new_files:
            continue

        with open(f"{articles_dir}/{article}") as f:
            data = json.load(f)

        nli_data = {
            "url": data["url"],
            "claim": data["claim"],
            "label": data["label"],
        }

        if len(data["sources"]) == 0:
            print("no sources", data["url"])
            continue

        new_sources = []
        entails = []
        neutrals = []
        contradictions = []
        for source in data["sources"]:
            source_text = source["text_cleaned"]
            if not source_text:
                source_text = source["text"]

            probs = model.get_probs(premise=source_text, hypothesis=data["claim"])
            new_sources.append(get_nli_source(source=source_text, probs=probs))

            entails.append(probs["entailment"])
            neutrals.append(probs["neutral"])
            contradictions.append(probs["contradiction"])

        nli_data["sources"] = new_sources

        stats = [
            min(entails),
            min(neutrals),
            min(contradictions),
            max(entails),
            max(neutrals),
            max(contradictions),
            mean(entails),
            mean(neutrals),
            mean(contradictions),
        ]

        nli_data["stats"] = stats

        with open(f"{new_articles_dir}/{article}", "w") as outfile:
            json.dump(nli_data, outfile, indent=4)


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


def load_data_from_urls(articles_dir: str, urls):
    infos = []
    for url in urls:
        article_filename = url.split("/")[-2]

        empty = [
            "nc-gov-roy-cooper-says-private-school-vouchers-lac",
            "city-council-chairwoman-leslie-curran-says-tarpon-",
        ]

        if article_filename in empty:
            continue

        with open(f"{articles_dir}/{article_filename}.json") as f:
            data = json.load(f)

        stats_dict = {
            STATS_LABELS[i]: data["stats"][i] for i in range(len(STATS_LABELS))
        }

        stats_dict["url"] = data["url"]
        stats_dict["label"] = data["label"]
        stats_dict["statsurl"] = data["stats"]

        infos.append(stats_dict)

    return infos


def load_data(urls_path: str, articles_dir: str):
    urls_test, urls_val, urls_train = load_datasplits_urls(urls_path=urls_path)

    test_data = load_data_from_urls(articles_dir=articles_dir, urls=urls_test)
    val_data = load_data_from_urls(articles_dir=articles_dir, urls=urls_val)
    train_data = load_data_from_urls(articles_dir=articles_dir, urls=urls_train)

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
    X_train, y_train = train_df[STATS_LABELS], train_df["label_encoded"]
    X_test, y_test = test_df[STATS_LABELS], test_df["label_encoded"]
    if validate:
        X_test, y_test = val_df[STATS_LABELS], val_df["label_encoded"]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    classification_report_dict = classification_report(
        y_test, y_pred, target_names=LABELS, output_dict=True
    )

    # print(classification_report(y_test, y_pred, target_names=labels))
    print("MAF1: ", classification_report_dict["macro avg"]["f1-score"])
    print("MAE: ", mae)
    print("MSE: ", mse)

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


def classification_by_nli_lr(articles_dir: str, validate=False, train_on_val=False):
    test_data, val_data, train_data = load_data(
        articles_dir=articles_dir, urls_path="./data/urls_split.json"
    )

    lr_model_args = {
        "max_iter": 400,
        "class_weight": "balanced",
        "C": 1.43,
    }
    model = LogisticRegression(**lr_model_args)

    print(lr_model_args)

    train_test_model(
        model=model,
        test_data=test_data,
        val_data=val_data,
        train_data=train_data,
        validate=validate,
        train_on_val=train_on_val,
        model_args=lr_model_args,
    )


def classification_by_nli_linear_svm(
    articles_dir: str, validate=False, train_on_val=False
):
    test_data, val_data, train_data = load_data(
        articles_dir=articles_dir, urls_path="./data/urls_split.json"
    )

    svm_model_args = {
        "penalty": "l2",  # default
        "max_iter": 1000,  # default
        "dual": False,
        "multi_class": "ovr",  # default
        "class_weight": "balanced",
        "C": 0.38,
        "loss": "squared_hinge",  # default
    }
    model = LinearSVC(**svm_model_args)

    print(svm_model_args)

    train_test_model(
        model=model,
        test_data=test_data,
        val_data=val_data,
        train_data=train_data,
        validate=validate,
        train_on_val=train_on_val,
        model_args=svm_model_args,
    )


def classification_by_nli_svm(articles_dir: str, validate=False, train_on_val=False):
    test_data, val_data, train_data = load_data(
        articles_dir=articles_dir, urls_path="./data/urls_split.json"
    )

    svm_model_args = {
        "max_iter": -1,  # default
        "class_weight": "balanced",
        "C": 1.475,
        "kernel": "poly",
        "degree": 15,
    }
    model = SVC(**svm_model_args)

    print(svm_model_args)

    train_test_model(
        model=model,
        test_data=test_data,
        val_data=val_data,
        train_data=train_data,
        validate=validate,
        train_on_val=train_on_val,
        model_args=svm_model_args,
    )


def main():
    # save_nli_probs(
    #     articles_dir="./data/articles_parsed_clean_date",
    #     new_articles_dir="./data/articles_nli_test",
    # )

    # classification_by_nli_lr(
    #     articles_dir="./data/articles_nli", validate=False, train_on_val=True
    # )

    # classification_by_nli_linear_svm(
    #     articles_dir="./data/articles_nli", validate=False, train_on_val=True
    # )

    # classification_by_nli_svm(
    #     articles_dir="./data/articles_nli", validate=False, train_on_val=True
    # )
    pass


if __name__ == "__main__":
    main()
