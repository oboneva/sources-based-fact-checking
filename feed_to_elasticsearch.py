import json
import time
from os import walk

import nltk
from elasticsearch import Elasticsearch, RequestsHttpConnection, helpers

es_client = Elasticsearch(
    "https://localhost:9200",
    connection_class=RequestsHttpConnection,
    http_auth=("elastic", "=fXuTGrhW8DtLlLEsa3w"),
    use_ssl=True,
    verify_certs=False,
)


def doc_generator(index: str, full_text: str):
    tokens = nltk.sent_tokenize(full_text)
    for idx, sentence in enumerate(tokens):
        yield {
            "_index": index,
            "_id": f"{idx}",
            "_source": {"sentence": sentence},
        }


def index_articles(articles_dir: str):
    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    for article in files:
        with open(f"{articles_dir}/{article}") as f:
            data = json.load(f)

        claim = data["claim"]

        i = 0
        for source in data["sources"]:
            for link in source["links"]:
                if "full_text_cleaned" in link and link["full_text_cleaned"] != "":
                    if "es-top4-hits" in link:
                        continue

                    article_id = article[:-5].lstrip("-").lower()
                    index = f"{article_id}-source{i}"

                    helpers.bulk(
                        es_client, doc_generator(index, link["full_text_cleaned"])
                    )

                    i += 1

                    time.sleep(0.75)

                    hits = search_phrase(index=index, phrase=claim)

                    hits = [hit["_source"]["sentence"] for hit in hits]

                    link["es-top4-hits"] = hits

                    es_client.indices.delete(index=index)

        with open(f"{articles_dir}/{article}", "w") as outfile:
            json.dump(data, outfile, indent=4)


def search_phrase(index, phrase):
    res = es_client.search(
        index=index, body={"query": {"match": {"sentence": phrase}}}, size=4
    )

    return res["hits"]["hits"]


def delete_all_indicies():
    indices = es_client.cat.indices(h="index").split()

    for index in indices:
        es_client.indices.delete(index=index)


def main():
    # delete_all_indicies()
    # index_articles(articles_dir="data/articles_parsed_with_full_text")
    pass


if __name__ == "__main__":
    main()
