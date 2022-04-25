import json
from datetime import datetime
from enum import Enum

import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
)
from torch.nn import L1Loss, MSELoss
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data_loading_utils import load_datasplits_urls
from metrics_constants import LABELS
from results_utils import save_conf_matrix


class EncodedInput(str, Enum):
    DOMAINS = "DOMAINS"
    TEXT = "TEXT"
    LINK_TEXT = "LINK_TEXT"
    LINK_TEXT_DOMAINS = "LINK_TEXT_DOMAINS"
    TRUNC_TO_TEXT = "TRUNC_TO_TEXT"


class Loss(str, Enum):
    CEL = "CEL"
    MAE = "MAE"
    MSE = "MSE"


class FCDataset(Dataset):
    def __init__(
        self,
        urls,
        articles_dir: str,
        encoded_input: EncodedInput,
        encode_author: bool,
        label2id,
        tokenizer,
        device,
    ):
        self.urls = urls
        self.articles_dir = articles_dir
        self.encoded_input = encoded_input
        self.encode_author = encode_author

        self.tokenizer = tokenizer
        self.label2id = label2id

        self.device = device

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, index):
        url = self.urls[index]

        article_filename = url.split("/")[-2]

        with open(f"{self.articles_dir}/{article_filename}.json") as f:
            data = json.load(f)

        label = data["label"]
        claim = data["claim"]
        author = data["author"]
        sources = []

        if self.encoded_input is EncodedInput.DOMAINS:
            for source in data["sources"]:
                for link in source["links"]:
                    sources.append(link["domain"])
        elif self.encoded_input is EncodedInput.TEXT:
            sources.extend(
                [
                    source["text_cleaned"] if source["text_cleaned"] else source["text"]
                    for source in data["sources"]
                ]
            )
        elif self.encoded_input is EncodedInput.LINK_TEXT:
            for source in data["sources"]:
                for link in source["links"]:
                    sources.append(link["link_text"])
        elif self.encoded_input is EncodedInput.LINK_TEXT_DOMAINS:
            for source in data["sources"]:
                for link in source["links"]:
                    sources.append(link["link_text"])
                    sources.append(link["domain"])
        elif self.encoded_input is EncodedInput.TRUNC_TO_LINK_TEXT:
            for source in data["sources"]:
                if len(source["links"]) == 0 or not source["links"][-1]["link_text"]:
                    continue
                last_link_text = source["links"][-1]["link_text"]
                parts = source["text"].split(last_link_text)
                trunc_source = parts[0] + " " + last_link_text
                sources.append(trunc_source)

        # encode target
        target = torch.zeros(len(self.label2id)).to(self.device)
        target[self.label2id[label]] = 1

        # enode domains
        texts_sep = " [SEP] ".join(sources)
        source_input = "[CLS] " + claim + " [SEP] " + texts_sep + " [SEP]"
        if self.encode_author:
            source_input = (
                "[CLS] " + author + " [SEP] " + claim + " [SEP] " + texts_sep + " [SEP]"
            )

        # tokenize input
        encoded_input = self.tokenizer(
            source_input, add_special_tokens=False, truncation=True
        )

        return {
            "input_ids": torch.tensor(encoded_input["input_ids"], device=self.device),
            "attention_mask": torch.tensor(
                encoded_input["attention_mask"], device=self.device
            ),
            "labels": target,
        }


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


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    labels = np.argmax(labels, axis=-1)

    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions, average="macro")
    recall = recall_score(y_true=labels, y_pred=predictions, average="macro")
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)

    return {"accuracy": accuracy, "recall": recall, "f1": f1, "mae": mae, "mse": mse}


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

    model_name = "distilbert-base-uncased"
    resume_from_checkpoint = ""
    freeze_base_model = True
    train_batch_size = 16
    encoded_input = EncodedInput.DOMAINS
    encode_author = True
    loss_func = Loss.MAE
    warmup = 0.1

    model_args = {
        "model_name": model_name,
        "freeze_base_model": freeze_base_model,
        "encoded_input": encoded_input,
        "loss_func": loss_func,
        "encode_author": encode_author,
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
        model_name, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    if freeze_base_model:
        for param in model.base_model.parameters():
            param.requires_grad = False

    model.to(device)

    # 2. Prepare the Trainer.
    model_save_name = "bert" if model_name == "distilbert-base-uncased" else "roberta"
    freeze_desc = "" if freeze_base_model else "_nofreeze"
    warmup_desc = "" if warmup == 0 else ("_warmup10" if warmup == 0.1 else "_warmup6")
    loss_desc = (
        "" if loss_func is Loss.CEL else ("_mae" if loss_func is Loss.MAE else "_mse")
    )
    input_desc = "_author+claim" if encode_author else "_claim_only"
    experiment_desc = f"bs{train_batch_size}_{model_save_name}{freeze_desc}{warmup_desc}{loss_desc}{input_desc}_{encoded_input}"
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
        compute_metrics=compute_metrics,
    )
    if loss_func is not Loss.CEL:
        trainer = CustomLossTrainer(
            **loss_func_param,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    predictions, label_ids, metrics = trainer.predict(test_dataset)
    print("Test: ", metrics)
    trainer.save_metrics("test", metrics)

    predictions = np.argmax(predictions, axis=-1)
    label_ids = np.argmax(label_ids, axis=-1)

    disp = ConfusionMatrixDisplay.from_predictions(
        label_ids, predictions, labels=[0, 1, 2, 3, 4, 5], display_labels=LABELS
    )

    save_conf_matrix(disp=disp, model_name=experiment_desc)

    params = {**train_args, **model_args}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    with open(f"./{output_dir}/model_params_{timestamp}.json", "w") as outfile:
        json.dump(params, outfile, indent=4)


if __name__ == "__main__":
    main()
