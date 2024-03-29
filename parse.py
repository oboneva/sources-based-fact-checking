import json

from bs_parser import BSParser


def add_title_to_parsed_data():
    urls_path = "./data/urls.json"
    articles_dir = "./data/articles"
    outdir = "./data/articles_parsed_clean_date"

    urls = []
    with open(urls_path) as f:
        data = json.load(f)
        urls.extend(data["urls"])

    for url in urls:
        filename = url.split("/")[-2]
        filepath = f"{articles_dir}/{filename}.html"

        parser = BSParser(filepath=filepath, url=url)
        title = parser.parse_title()

        ofile = filename.split(".")[0]
        with open(f"{outdir}/{ofile}.json", "r") as f:
            current_data = json.load(f)

        current_data["title"] = title
        with open(f"{outdir}/{ofile}.json", "w") as f:
            json.dump(current_data, f, indent=4)


def main():
    urls_path = "./data/urls.json"
    articles_dir = "./data/articles"
    outdir = "./data/articles_parsed"

    # archived = [
    #     "https://www.politifact.com/factchecks/2020/sep/16/li-meng-yan/tucker-carlson-guest-airs-debunked-conspiracy-theo/"
    # ]  # since it is only one, it is manually removed from all urls

    urls = []
    with open(urls_path) as f:
        data = json.load(f)
        urls.extend(data["urls"])

    for url in urls:
        # if url in archived:
        #     print("archived: ", url)
        #     continue
        filename = url.split("/")[-2]
        filepath = f"{articles_dir}/{filename}.html"

        parser = BSParser(filepath=filepath, url=url)
        data = parser.parse()

        ofile = filename.split(".")[0]
        with open(f"{outdir}/{ofile}.json", "w") as outfile:
            json.dump(data, outfile, indent=4)


if __name__ == "__main__":
    main()
