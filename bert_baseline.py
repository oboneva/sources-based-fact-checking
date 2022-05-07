import json
from datetime import datetime
from enum import Enum

import numpy as np
import torch
from coral_pytorch.dataset import proba_to_label
from sklearn.metrics import ConfusionMatrixDisplay
from torch.nn import L1Loss, MSELoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from compute_metrics import (
    compute_metrics_classification,
    compute_metrics_coral,
    compute_metrics_ordinal,
)
from custom_trainer.custom_loss_trainer import CustomLossTrainer
from custom_trainer.ordinal_regression_trainer import OrdinalRegressionTrainer
from data_loading_utils import load_datasplits_urls
from fc_dataset import EncodedInput, FCDataset
from metrics_constants import LABELS
from results_utils import save_conf_matrix
from roberta_coral_model import RobertaCoralForSequenceClassification


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
    ordinal_regression_coral = "ordinal_regression_coral"


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
    model_class = (
        RobertaCoralForSequenceClassification
        if task_type is TaskType.ordinal_regression_coral
        else AutoModelForSequenceClassification
    )
    model = model_class.from_pretrained(
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
    task_type_desc = ""
    if task_type is TaskType.ordinal_regression:
        task_type_desc = "_ord_reg"
    elif task_type is TaskType.ordinal_regression_coral:
        task_type_desc = "_coral"

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

    trainer_class = Trainer
    if task_type is TaskType.ordinal_regression:
        trainer_class = OrdinalRegressionTrainer
    elif loss_func is not Loss.CEL:
        trainer_class = CustomLossTrainer

    compute_metrics_func = compute_metrics_classification
    if task_type is TaskType.ordinal_regression:
        compute_metrics_func = compute_metrics_ordinal
    elif task_type is TaskType.ordinal_regression_coral:
        compute_metrics_func = compute_metrics_coral

    loss_func_param = {}
    if loss_func is not Loss.CEL:
        loss_func_param = {
            "loss_func": L1Loss() if loss_func is Loss.MAE else MSELoss()
        }

    trainer = trainer_class(
        **loss_func_param,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_func,
    )

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    predictions, label_ids, metrics = trainer.predict(test_dataset)
    print("Test: ", metrics)
    trainer.save_metrics("test", metrics)

    if task_type is TaskType.classification:
        predictions = np.argmax(predictions, axis=-1)
    elif task_type is TaskType.ordinal_regression:
        predictions = prediction2label(predictions)
    elif task_type is TaskType.ordinal_regression_coral:
        predictions = torch.sigmoid(torch.tensor(predictions))
        predictions = proba_to_label(predictions).float()

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
