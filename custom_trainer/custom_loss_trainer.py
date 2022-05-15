from transformers import Trainer

from bert_baseline import Loss
from torch.nn import L1Loss, MSELoss


class CustomLossTrainer(Trainer):
    def __init__(self, loss_type: Loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = L1Loss() if loss_type is Loss.MAE else MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss
        loss = self.loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss
