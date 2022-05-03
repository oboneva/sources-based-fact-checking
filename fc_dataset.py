import json
from enum import Enum

import torch
from torch.utils.data import Dataset


class EncodedInput(str, Enum):
    DOMAINS = "DOMAINS"
    TEXT = "TEXT"
    LINK_TEXT = "LINK_TEXT"
    LINK_TEXT_DOMAINS = "LINK_TEXT_DOMAINS"
    TRUNC_TO_LINK_TEXT = "TRUNC_TO_LINK_TEXT"


class FCDataset(Dataset):
    def __init__(
        self,
        urls,
        articles_dir: str,
        encoded_input: EncodedInput,
        encode_author: bool,
        label2id,
        tokenizer,
        device,
    ):
        self.urls = urls
        self.articles_dir = articles_dir
        self.encoded_input = encoded_input
        self.encode_author = encode_author

        self.tokenizer = tokenizer
        self.label2id = label2id

        self.device = device

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, index):
        url = self.urls[index]

        article_filename = url.split("/")[-2]

        with open(f"{self.articles_dir}/{article_filename}.json") as f:
            data = json.load(f)

        label = data["label"]
        claim = data["claim"]
        author = data["author"]
        sources = []

        if self.encoded_input is EncodedInput.DOMAINS:
            for source in data["sources"]:
                for link in source["links"]:
                    sources.append(link["domain"])
        elif self.encoded_input is EncodedInput.TEXT:
            sources.extend(
                [
                    source["text_cleaned"] if source["text_cleaned"] else source["text"]
                    for source in data["sources"]
                ]
            )
        elif self.encoded_input is EncodedInput.LINK_TEXT:
            for source in data["sources"]:
                for link in source["links"]:
                    sources.append(link["link_text"])
        elif self.encoded_input is EncodedInput.LINK_TEXT_DOMAINS:
            for source in data["sources"]:
                for link in source["links"]:
                    sources.append(link["link_text"])
                    sources.append(link["domain"])
        elif self.encoded_input is EncodedInput.TRUNC_TO_LINK_TEXT:
            for source in data["sources"]:
                if len(source["links"]) == 0 or not source["links"][-1]["link_text"]:
                    continue
                last_link_text = source["links"][-1]["link_text"]
                parts = source["text"].split(last_link_text)
                trunc_source = parts[0] + " " + last_link_text
                sources.append(trunc_source)

        # encode target
        target = torch.zeros(len(self.label2id)).to(self.device)
        target[self.label2id[label]] = 1

        # enode domains
        texts_sep = " [SEP] ".join(sources)
        source_input = "[CLS] " + claim + " [SEP] " + texts_sep + " [SEP]"
        if self.encode_author:
            source_input = (
                "[CLS] " + author + " [SEP] " + claim + " [SEP] " + texts_sep + " [SEP]"
            )

        # tokenize input
        encoded_input = self.tokenizer(
            source_input, add_special_tokens=False, truncation=True
        )

        return {
            "input_ids": torch.tensor(encoded_input["input_ids"], device=self.device),
            "attention_mask": torch.tensor(
                encoded_input["attention_mask"], device=self.device
            ),
            "labels": target,
        }
