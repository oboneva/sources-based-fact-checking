import argparse
import json
import math
import shutil
from os import walk

import dateparser

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


def get_flip_o_meter_urls(articles_dir):
    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    flip_o_meter_urls = []

    for article in files:
        if article == ".DS_Store":
            continue

        with open(f"{articles_dir}/{article}") as f:
            data = json.load(f)
            label = data["label"]

        if label in ["full-flop", "half-flip", "no-flip"]:
            flip_o_meter_urls.append(data["url"])

    return flip_o_meter_urls


def move_flip_o_meter_urls(urls, all_urls_path, new_file_path):
    all_urls = []
    with open(all_urls_path) as f:
        data = json.load(f)
        all_urls.extend(data["urls"])

    cleared_urls = [url for url in all_urls if url not in urls]

    with open(new_file_path, "w") as outfile:
        json.dump({"urls": cleared_urls}, outfile, indent=4)


def move_flip_o_meter_articles_parsed(urls, old_dir, new_dir):
    for url in urls:
        filename = url.split("/")[-2]
        shutil.move(f"{old_dir}/{filename}.json", f"{new_dir}/{filename}.json")


def move_flip_o_meter_articles_html(urls, old_dir, new_dir):
    for url in urls:
        filename = url.split("/")[-2]
        shutil.move(f"{old_dir}/{filename}.html", f"{new_dir}/{filename}.html")


def clear_flip_o_meter_articles():
    # urls = get_flip_o_meter_urls(articles_dir="./data/articles_parsed")

    # with open("./data/flip_o_meter_urls.json", "w") as outfile:
    #     json.dump({"urls": urls}, outfile, indent=4)

    urls = []
    with open("./data/flip_o_meter_urls.json") as f:
        data = json.load(f)
        urls.extend(data["urls"])

    move_flip_o_meter_urls(
        urls=urls,
        all_urls_path="./data/urls.json",
        new_file_path="./data/cleared_urls.json",
    )

    move_flip_o_meter_articles_html(
        urls=urls,
        old_dir="./data/articles",
        new_dir="./data/flip_o_meter_articles",
    )

    move_flip_o_meter_articles_parsed(
        urls=urls,
        old_dir="./data/articles_parsed",
        new_dir="./data/flip_o_meter_articles_parsed",
    )


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

    urls_splitted = {"test": test, "dev": dev, "train": train}

    print(" ---------------- Stats ---------------- ")
    print("Test: ", len(test))
    print("Dev: ", len(dev))
    print("Train: ", len(train))

    with open("./data/urls_splitted.json", "w") as outfile:
        json.dump(urls_splitted, outfile, indent=4)


def main():
    pass


if __name__ == "__main__":
    main()
