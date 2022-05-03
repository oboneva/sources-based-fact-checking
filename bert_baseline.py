import json
from datetime import datetime
from enum import Enum

import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.nn import L1Loss, MSELoss, Sigmoid
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from compute_metrics import compute_metrics_classification, compute_metrics_ordinal
from data_loading_utils import load_datasplits_urls
from fc_dataset import EncodedInput, FCDataset
from metrics_constants import LABELS
from results_utils import save_conf_matrix


class Loss(str, Enum):
    CEL = "CEL"
    MAE = "MAE"
    MSE = "MSE"


class ModelType(str, Enum):
    distil_roberta = "distilroberta-base"
    distil_bert = "distilbert-base-uncased"


class TaskType(str, Enum):
    classification = "classification"
    ordinal_regression = "ordinal_regression"


class OrdinalRegressionTrainer(Trainer):
    def __init__(self, loss_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.sigmoid = Sigmoid()

    def ordinal_regression(self, predictions, targets):
        modified_target = torch.zeros_like(predictions)

        for i, target in enumerate(targets):
            modified_target[i, 0 : target + 1] = 1

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

        probabilities = self.sigmoid(
            logits
        )  # e.g. [[0.5078, 0.5173, 0.4948, 0.4857, 0.4655, 0.5085], [0.4926, 0.5244, 0.4479, 0.5175, 0.4675, 0.5201]]

        labels = torch.argmax(labels, dim=1)  # convert [0, 0, 1, 0, 0, 0] to 2

        loss = self.ordinal_regression(probabilities, labels)

        return (loss, outputs) if return_outputs else loss


class CustomLossTrainer(Trainer):
    def __init__(self, loss_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss
        loss = self.loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss


def prediction2label(pred):
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    label2id = {LABELS[i]: i for i in range(len(LABELS))}
    id2label = {id: label for label, id in label2id.items()}
    num_labels = len(LABELS)

    # 1. Prepare the Data.
    urls_test, urls_val, urls_train = load_datasplits_urls(
        urls_path="data/urls_split_stratified.json"
    )
    aticles_dir = "data/articles_parsed_clean_date"

    task_type = TaskType.ordinal_regression
    model_name = ModelType.distil_roberta.value
    resume_from_checkpoint = ""
    freeze_base_model = False
    train_batch_size = 32
    encoded_input = EncodedInput.TEXT
    encode_author = False
    loss_func = Loss.MSE
    warmup = 0.06

    model_args = {
        "model_name": model_name,
        "freeze_base_model": freeze_base_model,
        "encoded_input": encoded_input,
        "loss_func": loss_func,
        "encode_author": encode_author,
        "task_type": task_type,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = FCDataset(
        urls=urls_train,
        articles_dir=aticles_dir,
        encoded_input=encoded_input,
        encode_author=encode_author,
        label2id=label2id,
        tokenizer=tokenizer,
        device=device,
    )
    val_dataset = FCDataset(
        urls=urls_val,
        articles_dir=aticles_dir,
        encoded_input=encoded_input,
        encode_author=encode_author,
        label2id=label2id,
        tokenizer=tokenizer,
        device=device,
    )
    test_dataset = FCDataset(
        urls=urls_test,
        articles_dir=aticles_dir,
        encoded_input=encoded_input,
        encode_author=encode_author,
        label2id=label2id,
        tokenizer=tokenizer,
        device=device,
    )

    # 2. Define the Model.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    if freeze_base_model:
        for param in model.base_model.parameters():
            param.requires_grad = False

    model.to(device)

    # 2. Prepare the Trainer.
    task_type_desc = "_ord_reg" if task_type is TaskType.ordinal_regression else ""
    model_save_name = "bert" if model_name == "distilbert-base-uncased" else "roberta"
    freeze_desc = "" if freeze_base_model else "_nofreeze"
    warmup_desc = "" if warmup == 0 else ("_warmup10" if warmup == 0.1 else "_warmup6")
    loss_desc = (
        "" if loss_func is Loss.CEL else ("_mae" if loss_func is Loss.MAE else "_mse")
    )
    input_desc = "_author+claim" if encode_author else "_claim_only"
    experiment_desc = f"bs{train_batch_size}_{model_save_name}{freeze_desc}{warmup_desc}{loss_desc}{input_desc}_{encoded_input}{task_type_desc}"
    output_dir = f"./output_{experiment_desc}"

    train_args = {
        "output_dir": output_dir,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "logging_strategy": "steps",
        "save_strategy": "epoch",
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 10,
        "logging_steps": 100,  # for loss, lr, epoch
        "eval_steps": 100,  # for compute metrics
        "report_to": "tensorboard",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "warmup_ratio": warmup,
        "resume_from_checkpoint": resume_from_checkpoint
        if resume_from_checkpoint
        else None,
    }

    training_args = TrainingArguments(**train_args)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    loss_func_param = {"loss_func": L1Loss() if loss_func is Loss.MAE else MSELoss()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_classification,
    )
    if loss_func is not Loss.CEL:
        trainer_class = (
            OrdinalRegressionTrainer
            if task_type is TaskType.ordinal_regression
            else CustomLossTrainer
        )

        trainer = trainer_class(
            **loss_func_param,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_ordinal
            if task_type is TaskType.ordinal_regression
            else compute_metrics_classification,
        )

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    predictions, label_ids, metrics = trainer.predict(test_dataset)
    print("Test: ", metrics)
    trainer.save_metrics("test", metrics)

    predictions = (
        prediction2label(predictions)
        if task_type is TaskType.ordinal_regression
        else np.argmax(predictions, axis=-1)
    )
    label_ids = np.argmax(label_ids, axis=-1)

    disp = ConfusionMatrixDisplay.from_predictions(
        label_ids,
        predictions,
        labels=[0, 1, 2, 3, 4, 5],
        display_labels=LABELS,
    )

    save_conf_matrix(disp=disp, model_name=experiment_desc)

    params = {**train_args, **model_args}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    with open(f"./{output_dir}/model_params_{timestamp}.json", "w") as outfile:
        json.dump(params, outfile, indent=4)


if __name__ == "__main__":
    main()
