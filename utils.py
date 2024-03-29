import argparse
import csv
import json
import math
from os import walk

import dateparser
import numpy as np

from data_loading_utils import load_datasplits_urls
from labels_mapping_utils import create_label2id_mapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-input_path", type=str)
parser.add_argument("-output_path", type=str)


def extract_urls():
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    files = []
    for (dirpath, dirnames, filenames) in walk(input_path):
        files.extend(filenames)
        break

    data = []
    for filename in files:
        with open(f"{input_path}/{filename}") as f:
            item = json.load(f)
            url = f'https://www.politifact.com{item["url"]}'
            data.append(url)

    with open(f"{output_path}/urls.json", "w", encoding="utf8") as outfile:
        json.dump({"urls": data}, outfile, indent=4)


def duplicates():
    urls_path = "./data/urls.json"

    urls = []
    with open(urls_path) as f:
        data = json.load(f)
        urls.extend(data["urls"])

    pages = []
    duplicated_urls = []
    for url in urls:
        page = url.split("/")[-2]
        if page in pages:
            duplicated_urls.append(url)
        pages.append(page)

    print(duplicated_urls)

    pages_unique = len(set(pages))
    print("pages_unique", pages_unique)

    pages_all = len(pages)
    print("pages_all", pages_all)

    if pages_unique == pages_all:
        print("All elements are unique.")
    else:
        print("All elements are not unique.")


def extract_unique_urls():
    urls = []
    with open("./data/urls.json") as f:
        data = json.load(f)
        urls.extend(data["urls"])

    unique_urls = []
    pages = []
    for url in urls:
        page = url.split("/")[-2]
        if page not in pages:
            unique_urls.append(url)
        pages.append(page)

    print("Unique urls: ", len(unique_urls))

    with open("./data/unique_urls.json", "w") as outfile:
        json.dump({"urls": unique_urls}, outfile, indent=4)


def create_data_splits_by_date():
    urls = []
    with open("./data/unique_urls.json") as f:
        data = json.load(f)
        urls.extend(data["urls"])

    urls_with_dates = []

    for url in urls:
        urls_parts = url.split("/")

        date_string = " ".join([urls_parts[-4], urls_parts[-5], urls_parts[-6]])
        date = dateparser.parse(date_string)

        urls_with_dates.append({"url": url, "date": date})

    urls_with_dates.sort(key=lambda x: x["date"], reverse=True)

    all_urls = len(urls_with_dates)

    test_count = math.ceil(all_urls * 0.2)
    dev_count = math.ceil(all_urls * 0.1)
    test = [urls_with_dates[i]["url"] for i in range(test_count)]
    dev = [urls_with_dates[i]["url"] for i in range(test_count, test_count + dev_count)]
    train = [urls_with_dates[i]["url"] for i in range(test_count + dev_count, all_urls)]

    urls_split = {"test": test, "dev": dev, "train": train}

    print(" ---------------- Stats ---------------- ")
    print("Test: ", len(test))
    print("Dev: ", len(dev))
    print("Train: ", len(train))

    with open("./data/urls_split.json", "w") as outfile:
        json.dump(urls_split, outfile, indent=4)


def compute_class_weights(articles_dir: str, num_classes: int):
    _, _, urls_train = load_datasplits_urls(urls_path="data/urls_split_stratified.json")

    label2id = create_label2id_mapper(num_classes=num_classes)
    print(label2id)
    counts = [0 for _ in range(num_classes)]

    for url in urls_train:
        article_filename = url.split("/")[-2]

        with open(f"{articles_dir}/{article_filename}.json") as f:
            data = json.load(f)

        label = data["label"]
        counts[label2id[label]] += 1

    print(counts)
    return min(counts) / np.array(counts)


def get_results(results_dir: str):
    folders = []
    for (dirpath, dirnames, filenames) in walk(results_dir):
        folders.extend(dirnames)
        break

    accs = []
    f1s = []
    recalls = []
    maes = []
    mses = []

    for folder in folders:
        with open(f"{results_dir}/{folder}/test_results.json") as f:
            data = json.load(f)
            accs.append(round(data["test_accuracy"], 5) * 100)
            f1s.append(round(data["test_f1"], 5) * 100)
            recalls.append(round(data["test_recall"], 5) * 100)

            maes.append(round(data["test_mae"], 3))
            mses.append(round(data["test_mse"], 3))

    zipofalllists = zip(folders, accs, f1s, recalls, maes, mses)
    output_columns = ["model", "acc", "f1", "recall", "mae", "mse"]
    with open(f"./{results_dir}/all_test_results.tsv", "w", newline="") as f_output:
        tsv_output = csv.writer(f_output, delimiter="\t")
        tsv_output.writerow(output_columns)
        for a, b, c, d, e, f in zipofalllists:
            tsv_output.writerow([a, b, c, d, e, f])


def main():
    # get_results(results_dir="results/...")
    # compute_class_weights(articles_dir="./data/articles_parsed_clean_date")
    pass


if __name__ == "__main__":
    main()
