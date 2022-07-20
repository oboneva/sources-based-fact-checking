import json
from os import walk

import trafilatura


def extract_full_text(articles_dir: str, destination_dir: str):
    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    results = []
    for (dirpath, dirnames, filenames) in walk(destination_dir):
        results.extend(filenames)
        break

    for article in files:
        if article == ".DS_Store":
            continue

        if article in results:
            continue

        with open(f"{articles_dir}/{article}") as f:
            data = json.load(f)

        for source in data["sources"]:
            for link_data in source["links"]:

                link = link_data["link"]

                if "www.timpawlenty.com" in link:
                    link_data["full_text"] = ""
                    continue

                page = trafilatura.fetch_url(link)

                if page is None:
                    link_data["full_text"] = ""
                else:
                    text = trafilatura.extract(page)
                    link_data["full_text"] = text

        with open(f"{destination_dir}/{article}", "w") as outfile:
            json.dump(data, outfile, indent=4)


def main():
    # extract_full_text(
    #     articles_dir="data/articles_parsed_clean_date", destination_dir="data/merged"
    # )
    pass


if __name__ == "__main__":
    main()
