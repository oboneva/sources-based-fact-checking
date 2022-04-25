import json
from os import walk

import dateparser
from unidecode import unidecode


def clean_access_dates(articles_dir: str):
    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    for article in files:
        if article == ".DS_Store":
            continue

        with open(f"{articles_dir}/{article}") as f:
            data = json.load(f)

        for source in data["sources"]:
            source_text = source["text"]

            source_text_parts = source_text.split(",")

            if len(source_text_parts) > 2:
                date_parts = source_text_parts[-2:]
                date_string = ",".join(date_parts)

                date = dateparser.parse(date_string)

                if date is not None:
                    cleaned_text = ",".join(
                        source_text_parts[: len(source_text_parts) - 2]
                    )
                    source["text_cleaned"] = cleaned_text
                    print("removes date", date_string)
                else:
                    source["text_cleaned"] = ""
                    # print("not date", date_string)
            else:
                source["text_cleaned"] = ""

        with open(f"{articles_dir}/{article}", "w") as outfile:
            json.dump(data, outfile, indent=4)


def clean_newlines(articles_dir: str):
    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    for article in files:
        if article == ".DS_Store":
            continue

        with open(f"{articles_dir}/{article}") as f:
            data = json.load(f)

        for source in data["sources"]:
            text_to_clean = (
                source["text_cleaned"] if source["text_cleaned"] else source["text"]
            )
            text_to_clean = text_to_clean.strip()

            text_to_clean = text_to_clean.replace("\n\n,", ",")
            text_to_clean = text_to_clean.replace("\n \n,", ",")

            text_to_clean = text_to_clean.replace(",\n\n", ", ")
            text_to_clean = text_to_clean.replace(",\n \n", ", ")

            text_to_clean = text_to_clean.replace(" \n\n", " ")
            text_to_clean = text_to_clean.replace("\n\n ", " ")

            text_to_clean = text_to_clean.replace("\n\n", " ")

            text_to_clean = text_to_clean.replace(" \n", " ")
            text_to_clean = text_to_clean.replace("\n ", " ")
            text_to_clean = text_to_clean.replace("\n", " ")

            source["text_cleaned"] = text_to_clean

            for link in source["links"]:
                text_link_to_clean = link["link_text"]

                text_link_to_clean = text_link_to_clean.strip()

                text_link_to_clean = text_link_to_clean.replace(" \n", " ")
                text_link_to_clean = text_link_to_clean.replace("\n ", " ")
                text_link_to_clean = text_link_to_clean.replace("\n", " ")

                link["link_text"] = text_link_to_clean

        with open(f"{articles_dir}/{article}", "w") as outfile:
            json.dump(data, outfile, indent=4)


def clean_empty_links(articles_dir: str):
    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    for article in files:
        if article == ".DS_Store":
            continue

        with open(f"{articles_dir}/{article}") as f:
            data = json.load(f)

        for source in data["sources"]:
            if (
                len(source["links"]) == 1
                and source["links"][0]["link"] == ""
                and source["links"][0]["link_text"] == ""
                and source["links"][0]["domain"] == ""
            ):
                source["links"] = []

        with open(f"{articles_dir}/{article}", "w") as outfile:
            json.dump(data, outfile, indent=4)


def clean_title(articles_dir: str):
    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    for article in files:
        if article == ".DS_Store":
            continue

        with open(f"{articles_dir}/{article}") as f:
            data = json.load(f)

        data["title"] = unidecode(data["title"].strip())

        with open(f"{articles_dir}/{article}", "w") as outfile:
            json.dump(data, outfile, indent=4)


def main():
    clean_title(articles_dir="./data/articles_parsed_clean_date")


if __name__ == "__main__":
    main()
