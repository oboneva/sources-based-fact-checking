import json
from datetime import datetime
from typing import Dict, List

from sklearn.metrics import ConfusionMatrixDisplay

from compute_metrics import compute_metrics
from fc_dataset import EncodedInput
from labels_mapping_utils import create_id2id_mapper, get_labels
from predict import get_predictions
from results_utils import save_conf_matrix


def map_and_save_results(
    predictions,
    label_ids,
    labels_mapper: Dict[int, int],
    display_labels: List[str],
    model_name: str,
):
    predictions = [labels_mapper[pred] for pred in predictions]
    label_ids = [labels_mapper[label_id] for label_id in label_ids]

    results = compute_metrics(label_ids, predictions)
    print(results)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    with open(
        f"results_{model_name}_{timestamp}.json",
        "w",
    ) as outfile:
        json.dump(results, outfile, indent=4)

    disp = ConfusionMatrixDisplay.from_predictions(
        label_ids,
        predictions,
        labels=sorted(set(labels_mapper.values())),
        display_labels=display_labels,
    )

    save_conf_matrix(disp=disp, model_name=model_name)


def main():
    model_name = ""
    predictions, label_ids = get_predictions(
        reverse_labels=False,
        ordinal=True,
        encoded_input=EncodedInput.TEXT,
        encode_author=True,
        model_checkpoint="",
    )

    predictions = predictions.numpy()

    for num_classes in [2, 3, 4]:
        labels_mapper = create_id2id_mapper(num_classes=num_classes)

        print(labels_mapper)

        map_and_save_results(
            predictions=predictions,
            label_ids=label_ids,
            labels_mapper=labels_mapper,
            display_labels=get_labels(num_classes=num_classes),
            model_name=f"{model_name}_{num_classes}",
        )


if __name__ == "__main__":
    main()
