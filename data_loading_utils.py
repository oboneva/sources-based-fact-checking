import json
from typing import List, Tuple


def load_datasplits_urls(urls_path: str) -> Tuple[List[str], List[str], List[str]]:
    urls_test = []
    urls_val = []
    urls_train = []
    with open(urls_path) as f:
        data = json.load(f)
        urls_test.extend(data["test"])
        urls_val.extend(data["dev"])
        urls_train.extend(data["train"])

    return urls_test, urls_val, urls_train
