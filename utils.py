import argparse
import json
from os import walk

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


def main():
    pass


if __name__ == "__main__":
    main()
