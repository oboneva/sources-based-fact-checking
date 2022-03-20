import json
from dataclasses import dataclass
from os import walk
from statistics import mean
import torch

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)

from nli_model import NLIModel


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


def save_nli_probs(articles_dir: str, new_articles_dir: str, device):
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

            probs = model.get_probs(premise=data["claim"], hypothesis=source_text)
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


def save_conf_matrix(disp, model_name: str):
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"{model_name}")
    plt.tight_layout()

    plt.savefig(
        f"conf_matrix_{model_name.lower()}.png",
        pad_inches=5,
        dpi=300,
    )


def save_model_stats(acc, mae, mse, report, model_name: str):
    results = {
        "name": model_name,
        "accuracie": acc,
        "mae": mae,
        "mse": mse,
        "classification_report": report,
    }

    with open(f"results_{model_name.lower()}.json", "w") as outfile:
        json.dump(results, outfile, indent=4)


def load_data_from_urls(articles_dir: str, urls):
    infos = []
    for url in urls:
        article_filename = url.split("/")[-2]

        with open(f"{articles_dir}/{article_filename}.json") as f:
            data = json.load(f)

        infos.append(
            {
                "url": data["url"],
                "label": data["label"],
                "stats": data["stats"],
            }
        )

    return infos


def load_data(urls_path: str, articles_dir: str):
    urls_test = []
    urls_val = []
    urls_train = []
    with open(urls_path) as f:
        data = json.load(f)
        urls_test.extend(data["test"])
        urls_val.extend(data["dev"])
        urls_train.extend(data["train"])

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


def train_test_model(model, test_data, val_data, train_data):
    test_df = pd.DataFrame(test_data)
    val_df = pd.DataFrame(val_data)
    train_df = pd.DataFrame(train_data)

    # encode labels
    labels = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    labels_mapper = {labels[i]: i + 1 for i in range(len(labels))}

    test_df = encode_label(test_df, labels_mapper=labels_mapper)
    val_df = encode_label(val_df, labels_mapper=labels_mapper)
    train_df = encode_label(train_df, labels_mapper=labels_mapper)

    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # separate x and y
    X_train, y_train = train_df["stats"], train_df["label_encoded"]
    X_test, y_test = test_df["stats"], test_df["label_encoded"]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    classification_report_dict = classification_report(
        y_test, y_pred, target_names=labels, output_dict=True
    )

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=[1, 2, 3, 4, 5, 6], display_labels=labels
    )

    model_name = type(model).__name__
    save_conf_matrix(disp=disp, model_name=model_name)

    save_model_stats(
        acc=acc,
        mae=mae,
        mse=mse,
        report=classification_report_dict,
        model_name=model_name,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    save_nli_probs(
        articles_dir="./data/articles_parsed_clean_date",
        new_articles_dir="./data/articles_nli_test",
        device=device
    )


if __name__ == "__main__":
    main()
