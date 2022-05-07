import numpy as np
import torch
from coral_pytorch.dataset import proba_to_label
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
)


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    recall = recall_score(y_true=y_true, y_pred=y_pred, average="macro")
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    return {"accuracy": accuracy, "recall": recall, "f1": f1, "mae": mae, "mse": mse}


def compute_metrics_ordinal(eval_preds):
    def prediction2label(pred):
        return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1

    logits, labels = eval_preds

    predictions = prediction2label(logits)
    labels = np.argmax(labels, axis=-1)

    return compute_metrics(y_true=labels, y_pred=predictions)


def compute_metrics_classification(eval_preds):
    logits, labels = eval_preds

    predictions = np.argmax(logits, axis=-1)
    labels = np.argmax(labels, axis=-1)

    return compute_metrics(y_true=labels, y_pred=predictions)


def compute_metrics_coral(eval_preds):
    logits, labels = eval_preds
    probas = torch.sigmoid(torch.tensor(logits))
    predictions = proba_to_label(probas).float()
    labels = np.argmax(labels, axis=-1)

    return compute_metrics(y_true=labels, y_pred=predictions)
