import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mord import LogisticAT
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.neural_network import MLPClassifier


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


def save_conf_matrix(disp, model_name: str, top_n_domains: int):
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"{model_name}, Top {top_n_domains} domains")
    plt.tight_layout()

    plt.savefig(
        f"conf_matrix_{model_name.lower()}_top_{top_n_domains}.png",
        pad_inches=5,
        dpi=300,
    )


def save_model_stats(top_n_domains, accs, maes, mses, model_name: str):
    fig, ax = plt.subplots(2, 2, constrained_layout=True)

    ax[0, 0].set_title("Accuracy")
    ax[0, 0].plot(top_n_domains, accs)
    ax[0, 0].set_xlabel("Top domains")
    ax[0, 0].set_ylabel("Accuracy score")

    ax[0, 1].set_title("MAE")
    ax[0, 1].plot(top_n_domains, maes)
    ax[0, 1].set_xlabel("Top domains")
    ax[0, 1].set_ylabel("MAE")

    ax[1, 0].set_title("MSE")
    ax[1, 0].plot(top_n_domains, mses)
    ax[1, 0].set_xlabel("Top domains")
    ax[1, 0].set_ylabel("MSE")

    ax[1, 1].set_visible(False)

    plt.suptitle(f"Stats for {model_name}")

    plt.savefig(
        f"stats_{model_name.lower()}.png",
        dpi=300,
    )

    results = {
        "name": model_name,
        "top_n_domains": top_n_domains,
        "accuracies": accs,
        "mae": maes,
        "mse": mses,
    }

    with open(f"results_{model_name.lower()}.json", "w") as outfile:
        json.dump(results, outfile, indent=4)


def train_test_model(
    model,
    top_n_domains: int,
    test_data,
    val_data,
    train_data,
    test_len,
    val_len,
    train_len,
):
    test_df = pd.DataFrame(test_data)
    val_df = pd.DataFrame(val_data)
    train_df = pd.DataFrame(train_data)

    # for this classification we are not using dev
    train_df = pd.concat([val_df, train_df], axis=0)

    all_data = pd.concat([test_df, train_df], axis=0)

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
    # val_df = all_data[test_len : test_len + val_len]
    train_df = all_data[test_len:]

    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # separate x and y
    X_train, y_train = train_df[domain_columns], train_df["label_encoded"]
    X_test, y_test = test_df[domain_columns], test_df["label_encoded"]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=[1, 2, 3, 4, 5, 6], display_labels=labels
    )

    save_conf_matrix(
        disp=disp, model_name=type(model).__name__, top_n_domains=top_n_domains
    )

    return acc, mae, mse


def classification_by_domain(articles_dir: str, top_n_domains, models):
    test_data, val_data, train_data, test_len, val_len, train_len = load_data(
        articles_dir=articles_dir, urls_path="./data/urls_split.json"
    )

    for model in models:
        accs = []
        maes = []
        mses = []

        for n_domains in top_n_domains:
            acc, mae, mse = train_test_model(
                model=model,
                top_n_domains=n_domains,
                test_data=test_data,
                val_data=val_data,
                train_data=train_data,
                test_len=test_len,
                val_len=val_len,
                train_len=train_len,
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
    top_n_domains = [i for i in range(250, 2750, 250)]

    models = [MLPClassifier(alpha=1, max_iter=1000), LogisticAT()]
    classification_by_domain(
        articles_dir="./data/articles_parsed",
        top_n_domains=top_n_domains,
        models=models,
    )


if __name__ == "__main__":
    main()
