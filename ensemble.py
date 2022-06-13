from typing import List

import numpy as np
import torch
from numpy import load
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import LinearSVR

from compute_metrics import compute_metrics
from fc_dataset import EncodedInput
from metrics_constants import LABELS
from predict import get_predictions
from results_utils import save_conf_matrix


def blending(models_dirs: List[str], meta_model):
    y_true_train = load(f"{models_dirs[0]}/predictions/y_true_train.npy")
    y_true = load(f"{models_dirs[0]}/predictions/y_true.npy")

    preds_type = "raw"

    concatenated = load(f"{models_dirs[0]}/predictions/{preds_type}_train.npy")
    concatenated_test = load(f"{models_dirs[0]}/predictions/{preds_type}.npy")

    for i in range(1, len(models_dirs)):
        raw_train = load(f"{models_dirs[i]}/predictions/{preds_type}_train.npy")
        raw = load(f"{models_dirs[i]}/predictions/{preds_type}.npy")

        concatenated = np.concatenate((concatenated, raw_train), axis=1)
        concatenated_test = np.concatenate((concatenated_test, raw), axis=1)

    meta_model.fit(concatenated, y_true_train)
    y_pred = meta_model.predict(concatenated_test)

    y_pred = [round(pred) for pred in y_pred]

    print(compute_metrics(y_true=y_true, y_pred=y_pred))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1, 2, 3, 4, 5], display_labels=LABELS
    )

    save_conf_matrix(disp=disp, model_name="")


def blending_ensemble():
    model_params = {
        "max_iter": 10000,
        "C": 0.114,
        "random_state": 2,
    }
    # print(model_params)

    model = LinearSVR(**model_params)

    blending(models_dirs=[""], meta_model=model)


def avg(predictions):
    predictions = torch.add(predictions[0], predictions[1])
    for i in range(2, len(predictions)):
        predictions = torch.add(predictions, predictions[i])

    predictions = torch.div(predictions, len(predictions))
    predictions = predictions.numpy()

    predictions = [round(pred) for pred in predictions]

    return predictions


def avg_ensemble():
    predictions1, _, label_ids = get_predictions(
        reverse_labels=False,
        ordinal=True,
        encoded_input=EncodedInput.TEXT,
        encode_author=False,
        model_checkpoint="",
    )
    predictions2, _, _ = get_predictions(
        reverse_labels=True,
        ordinal=True,
        encoded_input=EncodedInput.TEXT,
        encode_author=True,
        model_checkpoint="",
    )
    predictions3, _, _ = get_predictions(
        reverse_labels=True,
        ordinal=True,
        encoded_input=EncodedInput.TEXT,
        encode_author=True,
        model_checkpoint="",
    )

    predictions = avg(predictions=[predictions1, predictions2, predictions3])

    print(compute_metrics(label_ids, predictions))

    disp = ConfusionMatrixDisplay.from_predictions(
        label_ids, predictions, labels=[0, 1, 2, 3, 4, 5], display_labels=LABELS
    )

    save_conf_matrix(disp=disp, model_name="")


def main():
    pass


if __name__ == "__main__":
    main()