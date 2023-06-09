import collections
import json
from os import walk

import pdfplumber
import requests
from unidecode import unidecode

from preprocess_utils import clear_newlines


def extract_pdfs(articles_dir: str):
    extracted = 0

    not_working = []

    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    count = 0

    for article in files:
        if article == ".DS_Store":
            continue

        print(count)

        count += 1

        with open(f"{articles_dir}/{article}") as f:
            data = json.load(f)

        for source in data["sources"]:
            for link in source["links"]:
                if link["full_text"]:
                    continue

                url = link["link"]

                if not (url.endswith(".pdf") or url.endswith(".PDF")):
                    continue

                if url in not_working:
                    continue

                # print(url)

                try:
                    response = requests.get(url, timeout=(5, 60))
                except Exception as errt:
                    print(errt)
                    not_working.append(url)

                if (
                    response.status_code != 200
                    or "Content-Type" not in response.headers
                    or response.headers["Content-Type"] != "application/pdf"
                ):
                    not_working.append(url)
                    continue

                temp = open("temp.pdf", "wb")
                temp.write(response.content)
                temp.close()

                pdf = pdfplumber.open("temp.pdf")

                pdf_texts = []

                for pdf_page in pdf.pages[:100]:
                    pdf_texts.append(pdf_page.extract_text())

                pdf_text = " ".join(pdf_texts)

                link["full_text"] = pdf_text

                # extracted += 1
                # print("extracted", extracted)

        with open(f"{articles_dir}/{article}", "w") as outfile:
            json.dump(data, outfile, indent=4)

    with open("broken.json", "w") as outfile:
        json.dump({"not_working": not_working}, outfile, indent=4)


def stats_for_pdfs(articles_dir: str):
    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    pdf_urls = 0
    pdf_urls_with_any_text = 0

    texts = []
    duplicated = []

    for article in files:
        if article == ".DS_Store":
            continue

        with open(f"{articles_dir}/{article}") as f:
            data = json.load(f)

        for source in data["sources"]:
            for link in source["links"]:
                if "full_text_cleaned" in link:
                    continue

                url = link["link"]

                if url.endswith(".pdf") or url.endswith(".PDF"):
                    pdf_urls += 1

                    if (
                        "full_text" in link
                        and link["full_text"] is not None
                        and len(link["full_text"]) > 0
                    ):
                        pdf_urls_with_any_text += 1

                        if link["full_text"] in texts:
                            duplicated.append(link["full_text"])
                        else:
                            texts.append(link["full_text"])

    print("pdf_urls", pdf_urls)
    print("pdf_urls_with_any_text", pdf_urls_with_any_text)

    counter = collections.Counter(duplicated)
    print(sorted(list(set(counter.values()))))

    with open("duplicated_pdf_texts.json", "w") as outfile:
        json.dump({"duplicated": dict(counter)}, outfile, indent=4)


def clean_pdfs(articles_dir: str):
    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    pdf_urls = 0
    pdf_urls_with_any_text = 0

    for article in files:
        if article == ".DS_Store":
            continue

        with open(f"{articles_dir}/{article}") as f:
            data = json.load(f)

        for source in data["sources"]:
            for link in source["links"]:
                if "full_text_cleaned" in link:
                    continue

                url = link["link"]

                if not (url.endswith(".pdf") or url.endswith(".PDF")):
                    continue

                pdf_urls += 1

                if not (
                    "full_text" in link
                    and link["full_text"] is not None
                    and len(link["full_text"]) > 0
                ):
                    continue

                decoded = unidecode(link["full_text"].strip())

                cleaned = clear_newlines(text=decoded).strip()

                if len(cleaned) > 0:
                    pdf_urls_with_any_text += 1

                link["full_text_cleaned"] = cleaned

        with open(f"{articles_dir}/{article}", "w") as outfile:
            json.dump(data, outfile, indent=4)

    print("pdf_urls", pdf_urls)
    print("pdf_urls_with_any_text", pdf_urls_with_any_text)


def main():
    # extract_pdfs(articles_dir="data/articles_parsed_with_full_text")
    # stats_for_pdfs(articles_dir="data/articles_parsed_with_full_text")
    # clean_pdfs(articles_dir="data/articles_parsed_with_full_text")


if __name__ == "__main__":
    main()
