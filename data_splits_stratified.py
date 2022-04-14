import json

import pandas as pd
from sklearn.model_selection import train_test_split

from constants import LABELS


def create_data_splits_by_date(articles_dir: str):
    urls = []
    with open("./data/unique_urls.json") as f:
        data = json.load(f)
        urls.extend(data["urls"])

    url_label = []

    for url in urls:
        article_filename = url.split("/")[-2]

        with open(f"{articles_dir}/{article_filename}.json") as f:
            data = json.load(f)

        url_label.append({"url": url, "label": data["label"]})

    df = pd.DataFrame(url_label)

    labels_mapper = {LABELS[i]: i + 1 for i in range(len(LABELS))}
    df["label"] = df["label"].replace(labels_mapper)

    X_train, X_test, y_train, y_test = train_test_split(
        df["url"],
        df["label"],
        test_size=0.2,
        random_state=0,
        stratify=df["label"],
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.125,
        random_state=0,
        stratify=y_train,
    )

    urls_split = {
        "test": X_test.values.tolist(),
        "dev": X_val.values.tolist(),
        "train": X_train.values.tolist(),
    }

    with open("./data/urls_split_stratified.json", "w") as outfile:
        json.dump(urls_split, outfile, indent=4)


def main():
    create_data_splits_by_date(articles_dir="./data/articles_parsed")


if __name__ == "__main__":
    main()
