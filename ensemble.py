import torch
from sklearn.metrics import ConfusionMatrixDisplay

from compute_metrics import compute_metrics
from metrics_constants import LABELS
from predict import get_predictions
from results_utils import save_conf_matrix


def avg_ensemble():
    predictions1, label_ids = get_predictions(
        False,
        False,
        "output_bs32_roberta_nofreeze_warmup6_wcel_author+claim_TEXT/checkpoint-1365",
    )
    predictions2, _ = get_predictions(
        False,
        True,
        "output_bs32_roberta_nofreeze_warmup6_wmse_claim_only_TEXT_ord_reg/checkpoint-910",
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
