from functools import partial
from typing import List

import torch
from torch.nn import L1Loss, MSELoss, Sigmoid
from torchvision.ops import sigmoid_focal_loss
from transformers import Trainer

from .loss_type import Loss


class OrdinalRegressionTrainer(Trainer):
    def __init__(self, loss_type: Loss, weights: List[float], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights = torch.tensor(weights).to(self.device)

        if loss_type is Loss.MAE:
            self.loss_func = L1Loss(reduction="none")
        elif loss_type is Loss.MSE or loss_type is Loss.WMSE:
            self.loss_func = MSELoss(reduction="none")
        elif loss_type is Loss.FL:
            self.loss_func = partial(
                sigmoid_focal_loss, alpha=-1, gamma=2, reduction="mean"
            )

        self.loss_type = loss_type
        self.sigmoid = Sigmoid()

    def ordinal_regression(self, predictions, targets):
        modified_target = torch.zeros_like(predictions)

        for i, target in enumerate(targets):
            modified_target[i, 0 : target + 1] = 1

        if self.loss_type is Loss.MSE or self.loss_type is Loss.MAE:
            return torch.mean(
                torch.sum(self.loss_func(predictions, modified_target), dim=1)
            )
        elif self.loss_type is Loss.WMSE:
            weights = torch.tensor([self.weights[target] for target in targets]).to(
                self.device
            )
            return (
                torch.sum(self.loss_func(predictions, modified_target), dim=1) * weights
            ).sum() / weights.sum()

        return self.loss_func(predictions, modified_target)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get(
            "labels"
        )  # batch_size * 6, e.g. [[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get(
            "logits"
        )  # batch_size * 6, e.g. [[ 0.1385,  0.1279, -0.0427, -0.0823, -0.0810, -0.2731], [ 0.1227,  0.1638, -0.0378, -0.1190, -0.0404, -0.2168]]

        if self.loss_type is not Loss.FL:
            logits = self.sigmoid(
                logits
            )  # e.g. [[0.5078, 0.5173, 0.4948, 0.4857, 0.4655, 0.5085], [0.4926, 0.5244, 0.4479, 0.5175, 0.4675, 0.5201]]

        labels = torch.argmax(labels, dim=1)  # convert [0, 0, 1, 0, 0, 0] to 2

        loss = self.ordinal_regression(logits, labels)

        return (loss, outputs) if return_outputs else loss
