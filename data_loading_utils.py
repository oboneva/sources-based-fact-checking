import json
from typing import List, Tuple


def load_splitted_train_split(
    urls_path: str, ratio: float
) -> Tuple[List[str], List[str]]:
    urls_A = []
    urls_B = []

    ratio_desc = int(ratio * 100)

    with open(urls_path) as f:
        data = json.load(f)

        urls_A.extend(data[f"train{ratio_desc}"])
        urls_B.extend(data[f"train{100 - ratio_desc}"])

    return urls_A, urls_B  # 10, 90


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
