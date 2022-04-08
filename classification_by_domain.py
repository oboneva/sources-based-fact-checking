import json
from datetime import datetime

import numpy as np
import pandas as pd
from mord import LogisticAT
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)

from data_loading_utils import load_datasplits_urls
from results_utils import save_conf_matrix, save_model_stats

# from sklearn.neural_network import MLPClassifier


def load_data_from_urls(articles_dir: str, urls):
    infos = []
    for url in urls:
        article_filename = url.split("/")[-2]

        with open(f"{articles_dir}/{article_filename}.json") as f:
            data = json.load(f)

        for source in data["sources"]:
            if len(source["links"]) == 0:
                infos.append({"url": data["url"], "label": data["label"], "domain": ""})
            else:
                for link in source["links"]:
                    infos.append(
                        {
                            "url": data["url"],
                            "label": data["label"],
                            "domain": link["domain"],
                        }
                    )

    return infos


def load_data(articles_dir: str, urls_path: str):
    urls_test, urls_val, urls_train = load_datasplits_urls(urls_path=urls_path)

    test_data = load_data_from_urls(articles_dir=articles_dir, urls=urls_test)
    val_data = load_data_from_urls(articles_dir=articles_dir, urls=urls_val)
    train_data = load_data_from_urls(articles_dir=articles_dir, urls=urls_train)

    return (
        test_data,
        val_data,
        train_data,
        len(urls_test),
        len(urls_val),
        len(urls_train),
    )


def process_data(df, labels_mapper, top_n_domains: int, top_domains=None):
    df["domain"] = df["domain"].replace("", "none")

    df["label_encoded"] = df["label"].replace(labels_mapper)
    df = df.drop("label", 1)

    # get most popular domains
    if top_domains is None:
        top_domains = df["domain"].value_counts().index[: top_n_domains - 1].tolist()

    # assign 'other' to domains not incuded in top n
    df["top_domain"] = df["domain"].apply(lambda x: x if x in top_domains else "other")
    df = df.drop("domain", 1)

    # encode domains
    dummies = pd.get_dummies(df["top_domain"], prefix="domain")
    df = pd.concat([df, dummies], axis=1)
    df = df.drop("top_domain", 1)

    domain_columns = [f"domain_{domain}" for domain in top_domains]
    domain_columns.append("domain_other")

    df = df.groupby(["url", "label_encoded"])[domain_columns].agg("sum").reset_index()

    # normalize data
    df[domain_columns].apply(np.linalg.norm, axis=1)

    return domain_columns, df


def train_test_model(
    model,
    model_args,
    top_n_domains: int,
    test_data,
    val_data,
    train_data,
    test_len,
    val_len,
    train_len,
    validate: bool,
    train_on_val: bool,
):
    test_df = pd.DataFrame(test_data)
    val_df = pd.DataFrame(val_data)
    train_df = pd.DataFrame(train_data)

    all_data = pd.concat([test_df, val_df, train_df], axis=0)

    # encode labels
    labels = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    labels_mapper = {labels[i]: i + 1 for i in range(len(labels))}

    train_df["domain"] = train_df["domain"].replace("", "none")
    top_domains = train_df["domain"].value_counts().index[: top_n_domains - 1].tolist()

    domain_columns, all_data = process_data(
        all_data,
        labels_mapper=labels_mapper,
        top_n_domains=top_n_domains,
        top_domains=top_domains,
    )

    test_df = all_data[:test_len]
    val_df = all_data[test_len : test_len + val_len]
    train_df = all_data[test_len:]

    if train_on_val:
        train_df = pd.concat([val_df, train_df], axis=0)

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # separate x and y
    X_train, y_train = train_df[domain_columns], train_df["label_encoded"]
    X_test, y_test = test_df[domain_columns], test_df["label_encoded"]
    if validate:
        X_test, y_test = val_df[domain_columns], val_df["label_encoded"]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    classification_report_dict = classification_report(
        y_test, y_pred, target_names=labels, output_dict=True
    )

    dataset = "val" if validate else "test"

    results = {
        "name": type(model).__name__,
        "args": model_args,
        "top_n_domains": top_n_domains,
        "results_on": dataset,
        "mae": mae,
        "mse": mse,
        "classification_report": classification_report_dict,
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # args_string = "_".join([f"{k}_{v}" for k, v in model_args.items()])

    with open(
        f"results_{type(model).__name__.lower()}_td_{top_n_domains}_{dataset}_{timestamp}.json",
        "w",
    ) as outfile:
        json.dump(results, outfile, indent=4)

    print(classification_report(y_test, y_pred, target_names=labels))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=[1, 2, 3, 4, 5, 6], display_labels=labels
    )

    save_conf_matrix(
        disp=disp, model_name=type(model).__name__, top_n_domains=top_n_domains
    )

    return acc, mae, mse


def classification_by_domain(
    articles_dir: str, top_n_domains, model, model_args, validate, train_on_val=False
):
    test_data, val_data, train_data, test_len, val_len, train_len = load_data(
        articles_dir=articles_dir, urls_path="./data/urls_split.json"
    )

    accs = []
    maes = []
    mses = []

    for n_domains in top_n_domains:
        acc, mae, mse = train_test_model(
            model=model,
            model_args=model_args,
            top_n_domains=n_domains,
            test_data=test_data,
            val_data=val_data,
            train_data=train_data,
            test_len=test_len,
            val_len=val_len,
            train_len=train_len,
            validate=validate,
            train_on_val=train_on_val,
        )
        accs.append(acc)
        maes.append(mae)
        mses.append(mse)

    save_model_stats(
        top_n_domains=top_n_domains,
        accs=accs,
        maes=maes,
        mses=mses,
        model_name=type(model).__name__,
    )


def main():
    top_n_domains = [i for i in range(1000, 11000, 1000)]

    logistic_at_model_args = {"alpha": 0.5}
    logistic_at_model = LogisticAT(**logistic_at_model_args)

    # mlp_model_args = {"alpha": 1, "max_iter": 1000}
    # mlp_model = MLPClassifier(**mlp_model_args)

    classification_by_domain(
        articles_dir="./data/articles_parsed",
        top_n_domains=top_n_domains,
        model=logistic_at_model,
        model_args=logistic_at_model_args,
        validate=True,
        train_on_val=False,
    )


if __name__ == "__main__":
    main()
