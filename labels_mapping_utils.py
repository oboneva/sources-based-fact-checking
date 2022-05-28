from typing import Dict, List

from metrics_constants import LABELS


def get_labels(num_classes: int) -> List[str]:
    if num_classes == 2:
        return ["false", "true"]
    elif num_classes == 3:
        return ["false", "mixed", "true"]
    elif num_classes == 4:
        return ["pants-fire", "false", "mixed", "true"]

    return LABELS


def create_label2label_mapper(num_classes: int) -> Dict[str, str]:
    labels = get_labels(num_classes=num_classes)

    mapper = create_label2id_mapper(num_classes=num_classes)

    return {LABELS[i]: labels[mapper[LABELS[i]]] for i in range(len(LABELS))}


def create_label2id_mapper(num_classes: int) -> Dict[str, int]:
    all_classes = len(LABELS)

    if num_classes == 2:
        return {LABELS[i]: i // 3 for i in range(all_classes)}
    elif num_classes == 3:
        return {LABELS[i]: i // 2 for i in range(all_classes)}
    elif num_classes == 4:
        mapper = {LABELS[i]: (i // 2) + 1 for i in range(all_classes)}
        mapper["pants-fire"] = 0
        return mapper

    return {LABELS[i]: i for i in range(all_classes)}


def create_id2id_mapper(num_classes: int) -> Dict[int, int]:
    all_classes = len(LABELS)

    if num_classes == 2:
        return {i: i // 3 for i in range(all_classes)}
    elif num_classes == 3:
        return {i: i // 2 for i in range(all_classes)}
    elif num_classes == 4:
        mapper = {i: (i // 2) + 1 for i in range(all_classes)}
        mapper[0] = 0
        return mapper

    return {i: i for i in range(all_classes)}


def main():
    pass


if __name__ == "__main__":
    main()
