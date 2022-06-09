import csv
import json
from datetime import datetime

import numpy as np
import torch
from coral_pytorch.dataset import proba_to_label
from sklearn.metrics import ConfusionMatrixDisplay
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
from custom_trainer.loss_type import Loss
from custom_trainer.ordinal_regression_trainer import OrdinalRegressionTrainer
from data_loading_utils import load_datasplits_urls, load_splitted_train_split
from fc_dataset import EncodedInput, FCDataset
from labels_mapping_utils import create_label2id_mapper, get_labels, get_weights
from params_type import ModelType, TaskType
from results_utils import save_conf_matrix
from roberta_coral_model import RobertaCoralForSequenceClassification


def prediction2label(pred):
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    reverse_labels = False
    num_classes = 6
    train_on_all_train_data = True

    labels = get_labels(num_classes=num_classes)
    weights = get_weights(num_classes=num_classes)

    if reverse_labels:
        labels = labels[::-1]
        weights = weights[::-1]

    label2id = {labels[i]: i for i in range(len(labels))}
    id2label = {id: label for label, id in label2id.items()}
    num_labels = len(labels)

    # 1. Prepare the Data.
    urls_test, urls_val, urls_train = load_datasplits_urls(
        urls_path="data/urls_split_stratified.json"
    )

    urls_train_less, urls_train_more = load_splitted_train_split(
        urls_path="data/urls_train_split_90_10.json", ratio=0.1
    )

    if not train_on_all_train_data:
        urls_train = urls_train_more

    aticles_dir = "data/articles_parsed_clean_date"

    task_type = TaskType.ordinal_regression
    model_name = ModelType.distil_roberta.value
    resume_from_checkpoint = ""
    freeze_base_model = False
    train_batch_size = 32
    encoded_input = EncodedInput.TEXT
    encode_author = False
    loss_type = Loss.MSE
    warmup = 0.06

    model_args = {
        "model_name": model_name,
        "freeze_base_model": freeze_base_model,
        "encoded_input": encoded_input,
        "loss_func": loss_type,
        "encode_author": encode_author,
        "task_type": task_type,
        "reverse_labels": reverse_labels,
        "num_classes": num_classes,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    labels_mapper = create_label2id_mapper(num_classes=num_classes)
    train_dataset = FCDataset(
        urls=urls_train,
        articles_dir=aticles_dir,
        encoded_input=encoded_input,
        encode_author=encode_author,
        all_labels2id=labels_mapper,
        tokenizer=tokenizer,
        device=device,
    )
    val_dataset = FCDataset(
        urls=urls_val,
        articles_dir=aticles_dir,
        encoded_input=encoded_input,
        encode_author=encode_author,
        all_labels2id=labels_mapper,
        tokenizer=tokenizer,
        device=device,
    )
    test_dataset = FCDataset(
        urls=urls_test,
        articles_dir=aticles_dir,
        encoded_input=encoded_input,
        encode_author=encode_author,
        all_labels2id=labels_mapper,
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

    labels_desc = "_rev" if reverse_labels else ""
    model_save_name = "bert" if model_name == "distilbert-base-uncased" else "roberta"
    freeze_desc = "" if freeze_base_model else "_nofreeze"
    warmup_desc = "" if warmup == 0 else ("_warmup10" if warmup == 0.1 else "_warmup6")
    loss_desc = "" if loss_type is Loss.CEL else f"_{loss_type.value.lower()}"
    input_desc = "_author+claim" if encode_author else "_claim_only"
    experiment_desc = f"bs{train_batch_size}_{model_save_name}{freeze_desc}{warmup_desc}{loss_desc}{input_desc}_{encoded_input}{task_type_desc}{labels_desc}"
    output_dir = f"./output_{experiment_desc}"

    metric_for_best_model = "mae"
    greater_is_better = False

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
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
    }

    training_args = TrainingArguments(**train_args)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer_class = Trainer
    if task_type is TaskType.ordinal_regression:
        trainer_class = OrdinalRegressionTrainer
    elif loss_type is not Loss.CEL:
        trainer_class = CustomLossTrainer

    compute_metrics_func = compute_metrics_classification
    if task_type is TaskType.ordinal_regression:
        compute_metrics_func = compute_metrics_ordinal
    elif task_type is TaskType.ordinal_regression_coral:
        compute_metrics_func = compute_metrics_coral

    trainer_params = {}
    if loss_type is not Loss.CEL:
        trainer_params = {"loss_type": loss_type, "weights": weights}

    trainer = trainer_class(
        **trainer_params,
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

    raw_predictions = predictions
    if task_type is TaskType.classification:
        predictions = np.argmax(predictions, axis=-1)
    elif task_type is TaskType.ordinal_regression:
        raw_predictions = torch.sigmoid(torch.tensor(predictions))
        predictions = prediction2label(raw_predictions)
    elif task_type is TaskType.ordinal_regression_coral:
        raw_predictions = torch.sigmoid(torch.tensor(predictions))
        predictions = proba_to_label(raw_predictions).float()

    if task_type is not TaskType.classification:
        raw_predictions = raw_predictions.numpy()
        predictions = predictions.numpy()

    label_ids = np.argmax(label_ids, axis=-1)
    raw_predictions = [[round(a, 2) for a in pred] for pred in raw_predictions]

    zipofalllists = zip(raw_predictions, predictions, label_ids)
    output_columns = ["probabilities", "y_pred", "y_true"]
    with open(f"./{output_dir}/preds.tsv", "w", newline="") as f_output:
        tsv_output = csv.writer(f_output, delimiter="\t")
        tsv_output.writerow(output_columns)
        for a, b, c in zipofalllists:
            tsv_output.writerow([a, b, c])

    disp = ConfusionMatrixDisplay.from_predictions(
        label_ids,
        predictions,
        labels=sorted(set(labels_mapper.values())),
        display_labels=labels,
    )

    save_conf_matrix(disp=disp, model_name=experiment_desc)

    params = {**train_args, **model_args}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    with open(f"./{output_dir}/model_params_{timestamp}.json", "w") as outfile:
        json.dump(params, outfile, indent=4)


if __name__ == "__main__":
    main()
