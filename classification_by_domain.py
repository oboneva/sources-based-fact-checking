import json
from os import walk

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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def load_data(articles_dir: str):
    # load files info to a dataframe
    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    infos = []
    for article in files:
        if article == ".DS_Store":
            continue

        with open(f"{articles_dir}/{article}") as f:
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


def train_test_model(data, top_n_domains: int, model):
    df = pd.DataFrame(data)
    df["domain"] = df["domain"].replace("", "none")

    # encode labels
    labels = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    labels_mapper = {labels[i]: i + 1 for i in range(len(labels))}

    df["label_encoded"] = df["label"].replace(labels_mapper)
    df = df.drop("label", 1)

    # get most popular domains
    top_domains = df["domain"].value_counts().index[: top_n_domains - 1].tolist()

    # assign 'other' to domains not incuded in top n
    df["top_domain"] = df["domain"].apply(lambda x: x if x in top_domains else "other")
    df = df.drop("domain", 1)

    # encode domains
    df = pd.concat([df, pd.get_dummies(df["top_domain"], prefix="domain")], axis=1)
    df = df.drop("top_domain", 1)

    top_domains.append("other")
    domain_columns = [f"domain_{domain}" for domain in top_domains]

    df = df.groupby(["url", "label_encoded"])[domain_columns].agg("sum").reset_index()

    # normalize data
    df[domain_columns].apply(np.linalg.norm, axis=1)

    # separate datas
    X_train, X_test, y_train, y_test = train_test_split(
        df[domain_columns],
        df["label_encoded"],
        test_size=0.2,
        random_state=0,
        stratify=df["label_encoded"],
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    model_name = type(model).__name__

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=[1, 2, 3, 4, 5, 6], display_labels=labels
    )

    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"{model_name}, Top {top_n_domains} domains")
    plt.tight_layout()

    plt.savefig(
        f"conf_matrix_{model_name.lower()}_top_{top_n_domains}.png",
        pad_inches=5,
        dpi=300,
    )

    return acc, mae, mse


def classification_by_domain(articles_dir: str, top_n_domains, models):
    infos = load_data(articles_dir=articles_dir)

    for model in models:
        accs = []
        maes = []
        mses = []

        for n_domains in top_n_domains:
            acc, mae, mse = train_test_model(
                data=infos, top_n_domains=n_domains, model=model
            )
            accs.append(acc)
            maes.append(mae)
            mses.append(mse)

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

        model_name = type(model).__name__
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


def main():
    top_n_domains = [i for i in range(250, 2750, 250)]

    models = [MLPClassifier(alpha=1, max_iter=100), LogisticAT()]
    classification_by_domain(
        articles_dir="./data/articles_parsed",
        top_n_domains=top_n_domains,
        models=models,
    )


if __name__ == "__main__":
    main()
