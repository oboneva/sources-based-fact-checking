import json
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from data_loading_utils import load_datasplits_urls
from metrics_constants import LABELS


def create_url_label_dataframe(articles_dir: str, urls: List[str]) -> pd.DataFrame:
    url_label = []

    for url in urls:
        article_filename = url.split("/")[-2]

        with open(f"{articles_dir}/{article_filename}.json") as f:
            data = json.load(f)

        url_label.append({"url": url, "label": data["label"]})

    df = pd.DataFrame(url_label)

    labels_mapper = {LABELS[i]: i + 1 for i in range(len(LABELS))}
    df["label"] = df["label"].replace(labels_mapper)

    return df


def split_train(articles_dir: str, urls_path: str, ratio: float):
    _, _, urls_train = load_datasplits_urls(urls_path=urls_path)

    df = create_url_label_dataframe(articles_dir=articles_dir, urls=urls_train)

    X_train, X_test, y_train, y_test = train_test_split(
        df["url"],
        df["label"],
        test_size=ratio,
        random_state=0,
        stratify=df["label"],
    )

    ratio_desc = int(ratio * 100)

    urls_split = {
        f"train{ratio_desc}": X_test.values.tolist(),
        f"train{100 - ratio_desc}": X_train.values.tolist(),
    }

    with open(
        f"./data/urls_train_split_{100 - ratio_desc}_{ratio_desc}.json", "w"
    ) as outfile:
        json.dump(urls_split, outfile, indent=4)


def create_stratified_data_splits(articles_dir: str):
    urls = []
    with open("./data/unique_urls.json") as f:
        data = json.load(f)
        urls.extend(data["urls"])

    df = create_url_label_dataframe(articles_dir=articles_dir, urls=urls)

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
    # create_stratified_data_splits(articles_dir="./data/articles_parsed")
    # split_train(
    #     articles_dir="./data/articles_parsed",
    #     urls_path="./data/urls_split_stratified.json",
    #     ratio=0.1,
    # )
    pass


if __name__ == "__main__":
    main()
