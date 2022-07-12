from typing import List

import numpy as np
import torch
from mord import LogisticAT
from numpy import load
from sklearn.metrics import ConfusionMatrixDisplay

from compute_metrics import compute_metrics
from metrics_constants import LABELS
from predict import get_test_predictions
from results_utils import save_conf_matrix


def avg(predictions):
    predictions = torch.add(predictions[0], predictions[1])
    for i in range(2, len(predictions)):
        predictions = torch.add(predictions, predictions[i])

    predictions = torch.div(predictions, len(predictions))
    predictions = predictions.numpy()

    predictions = [round(pred) for pred in predictions]

    return predictions


def avg_ensemble():
    predictions1, _, label_ids = get_test_predictions(model_checkpoint="")
    predictions2, _, _ = get_test_predictions(model_checkpoint="")
    predictions3, _, _ = get_test_predictions(model_checkpoint="")

    predictions = avg(predictions=[predictions1, predictions2, predictions3])

    print(compute_metrics(label_ids, predictions))

    disp = ConfusionMatrixDisplay.from_predictions(
        label_ids, predictions, labels=[0, 1, 2, 3, 4, 5], display_labels=LABELS
    )

    save_conf_matrix(disp=disp, model_name="")


def train_eval_meta_model(models_dirs: List[str], meta_model, is_blending: bool):
    train = "train10" if is_blending else "train"
    test = "test"

    y_true_train = load(f"{models_dirs[0]}/predictions/{train}_y_true.npy")
    y_true_test = load(f"{models_dirs[0]}/predictions/{test}_y_true.npy")

    preds_type = "raw"

    concatenated = load(f"{models_dirs[0]}/predictions/{train}_{preds_type}.npy")
    concatenated_test = load(f"{models_dirs[0]}/predictions/{test}_{preds_type}.npy")

    for i in range(1, len(models_dirs)):
        raw_train = load(f"{models_dirs[i]}/predictions/{train}_{preds_type}.npy")
        raw = load(f"{models_dirs[i]}/predictions/{test}_{preds_type}.npy")

        concatenated = np.concatenate((concatenated, raw_train), axis=1)
        concatenated_test = np.concatenate((concatenated_test, raw), axis=1)

    meta_model.fit(concatenated, y_true_train)
    y_pred = meta_model.predict(concatenated_test)

    y_pred = [round(pred) for pred in y_pred]

    metrics = compute_metrics(y_true=y_true_test, y_pred=y_pred)

    # print(metrics)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true_test, y_pred, labels=[0, 1, 2, 3, 4, 5], display_labels=LABELS
    )

    model_name = f"{type(meta_model).__name__}_{len(models_dirs)}best"
    save_conf_matrix(disp=disp, model_name=model_name)

    return metrics["mae"]


def experiments_stacking():
    model_params = {
        "max_iter": 25000,
    }
    print(model_params)

    model = LogisticAT(**model_params)

    checkpoints = []

    _ = train_eval_meta_model(
        models_dirs=checkpoints, meta_model=model, is_blending=True
    )


def main():
    pass


if __name__ == "__main__":
    main()
