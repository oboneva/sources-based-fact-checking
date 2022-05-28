from typing import List

import torch
from bert_baseline import Loss
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from transformers import Trainer


class CustomLossTrainer(Trainer):
    def __init__(self, loss_type: Loss, weights: List[float], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = L1Loss() if loss_type is Loss.MAE else MSELoss()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights = torch.tensor(weights).to(device)

        if loss_type is Loss.WCEL:
            self.loss_func = CrossEntropyLoss(weight=self.weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss
        loss = self.loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss
