import json
from os import walk

import dateparser


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


def main():
    clean_access_dates(articles_dir="./data/articles_parsed_clean_date")


if __name__ == "__main__":
    main()
