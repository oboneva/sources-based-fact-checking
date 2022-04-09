import json

from data_loading_utils import load_datasplits_urls
from nli_baseline import STATS_LABELS


def load_and_convert_data_from_urls(articles_dir: str, urls):
    labels = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    labels_mapper = {labels[i]: i for i in range(len(labels))}

    data_lines = []
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

        features = [f'{i + 1}:{data["stats"][i]}' for i in range(len(STATS_LABELS))]
        features_str = " ".join(features)
        label_str = f'{labels_mapper[data["label"]]}'

        line = label_str + " " + features_str + "\n"

        data_lines.append(line)

    return data_lines


def convert_data_to_libsvm_format(urls_path: str, articles_dir: str):
    urls_test, urls_val, urls_train = load_datasplits_urls(urls_path=urls_path)

    test_data_lines = load_and_convert_data_from_urls(
        articles_dir=articles_dir, urls=urls_test
    )
    val_data_lines = load_and_convert_data_from_urls(
        articles_dir=articles_dir, urls=urls_val
    )
    train_data_lines = load_and_convert_data_from_urls(
        articles_dir=articles_dir, urls=urls_train
    )

    with open("test_data", "w") as f:
        f.writelines(test_data_lines)

    with open("val_data", "w") as f:
        f.writelines(val_data_lines)

    with open("train_data", "w") as f:
        f.writelines(train_data_lines)


def main():
    convert_data_to_libsvm_format(
        articles_dir="./data/articles_nli", urls_path="./data/urls_split.json"
    )


if __name__ == "__main__":
    main()
