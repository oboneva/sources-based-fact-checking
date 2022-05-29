import torch
from sklearn.metrics import ConfusionMatrixDisplay

from compute_metrics import compute_metrics
from fc_dataset import EncodedInput
from metrics_constants import LABELS
from predict import get_predictions
from results_utils import save_conf_matrix


def avg_ensemble():
    predictions1, label_ids = get_predictions(
        reverse_labels=False,
        ordinal=True,
        encoded_input=EncodedInput.TEXT,
        encode_author=True,
        model_checkpoint="",
    )
    predictions2, _ = get_predictions(
        reverse_labels=True,
        ordinal=True,
        encoded_input=EncodedInput.TEXT,
        encode_author=True,
        model_checkpoint="",
    )
    # predictions3, _ = get_predictions(False, False, "")

    predictions = torch.add(predictions1, predictions2)
    # predictions = torch.add(predictions, predictions3)

    predictions = torch.div(predictions, 2)
    predictions = predictions.numpy()

    predictions = [round(pred) for pred in predictions]

    print(compute_metrics(label_ids, predictions))

    disp = ConfusionMatrixDisplay.from_predictions(
        label_ids,
        predictions,
        labels=[0, 1, 2, 3, 4, 5],
        display_labels=LABELS,
    )

    save_conf_matrix(disp=disp, model_name="ensemble")


def main():
    pass


if __name__ == "__main__":
    main()
