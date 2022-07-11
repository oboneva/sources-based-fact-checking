import json
from dataclasses import dataclass
from os import walk
from statistics import mean

import torch

from nli_model import NLIModel


@dataclass
class NLISource:
    text: str
    entailment: float
    neutral: float
    contradiction: float


def get_nli_source(source: str, probs):
    return vars(
        NLISource(
            text=source,
            entailment=probs["entailment"],
            neutral=probs["neutral"],
            contradiction=probs["contradiction"],
        )
    )


def save_nli_probs(articles_dir: str, new_articles_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    files = []
    for (dirpath, dirnames, filenames) in walk(articles_dir):
        files.extend(filenames)
        break

    new_files = []
    for (dirpath, dirnames, filenames) in walk(new_articles_dir):
        new_files.extend(filenames)
        break

    model = NLIModel("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", device)

    for article in files:
        if article == ".DS_Store" or article in new_files:
            continue

        with open(f"{articles_dir}/{article}") as f:
            data = json.load(f)

        nli_data = {
            "url": data["url"],
            "claim": data["claim"],
            "label": data["label"],
        }

        if len(data["sources"]) == 0:
            print("no sources", data["url"])
            continue

        new_sources = []
        entails = []
        neutrals = []
        contradictions = []
        for source in data["sources"]:
            source_text = source["text_cleaned"]
            if not source_text:
                source_text = source["text"]

            probs = model.get_probs(premise=source_text, hypothesis=data["claim"])
            new_sources.append(get_nli_source(source=source_text, probs=probs))

            entails.append(probs["entailment"])
            neutrals.append(probs["neutral"])
            contradictions.append(probs["contradiction"])

        nli_data["sources"] = new_sources

        stats = [
            min(entails),
            min(neutrals),
            min(contradictions),
            max(entails),
            max(neutrals),
            max(contradictions),
            mean(entails),
            mean(neutrals),
            mean(contradictions),
        ]

        nli_data["stats"] = stats

        with open(f"{new_articles_dir}/{article}", "w") as outfile:
            json.dump(nli_data, outfile, indent=4)


def main():
    # save_nli_probs(
    #     articles_dir="./data/articles_parsed_clean_date",
    #     new_articles_dir="./data/articles_nli_test",
    # )

    pass


if __name__ == "__main__":
    main()
