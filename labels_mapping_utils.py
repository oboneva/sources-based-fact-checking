from typing import Dict, List

from metrics_constants import LABELS, WEIGHTS


def get_weights(num_classes: int) -> List[float]:
    if num_classes == 2:
        return [0.82592313, 1.0]
    elif num_classes == 3:
        return [0.7297781, 0.83702791, 1.0]
    elif num_classes == 4:
        return [1.0, 0.48498538, 0.37458949, 0.4475233]

    return WEIGHTS


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
