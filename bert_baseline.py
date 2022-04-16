import json

import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
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


class FCDataset(Dataset):
    def __init__(
        self, urls, articles_dir: str, encode_domains: bool, label2id, tokenizer, device
    ):
        self.urls = urls
        self.articles_dir = articles_dir
        self.encode_domains = encode_domains

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
        source_domains = []
        source_texts = []

        if self.encode_domains:
            for source in data["sources"]:
                for link in source["links"]:
                    source_domains.append(link["domain"])
        else:
            source_texts.extend(
                [
                    source["text_cleaned"] if source["text_cleaned"] else source["text"]
                    for source in data["sources"]
                ]
            )

        # encode target
        target = torch.zeros(len(self.label2id)).to(self.device)
        target[self.label2id[label]] = 1

        # enode domains
        texts_sep = " [SEP] ".join(
            source_domains if self.encode_domains else source_texts
        )
        source_input = "[CLS] " + claim + " [SEP] " + texts_sep + " [SEP]"

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


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    labels = np.argmax(labels, axis=-1)

    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions, average="macro")
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)

    return {"accuracy": accuracy, "f1": f1, "mae": mae, "mse": mse}


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

    encode_domains = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = FCDataset(
        urls=urls_train,
        articles_dir=aticles_dir,
        encode_domains=encode_domains,
        label2id=label2id,
        tokenizer=tokenizer,
        device=device,
    )
    val_dataset = FCDataset(
        urls=urls_val,
        articles_dir=aticles_dir,
        encode_domains=encode_domains,
        label2id=label2id,
        tokenizer=tokenizer,
        device=device,
    )
    test_dataset = FCDataset(
        urls=urls_test,
        articles_dir=aticles_dir,
        encode_domains=encode_domains,
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
    train_batch_size = 16
    input = "domains" if encode_domains else "text"
    model_save_name = "bert" if model_name == "distilbert-base-uncased" else "roberta"
    freeze_desc = "" if freeze_base_model else "_nofreeze"
    output_dir = f"./output_bs{train_batch_size}_{model_save_name}_{input}{freeze_desc}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        save_strategy="epoch",
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=128,
        num_train_epochs=60,
        logging_steps=100,  # for loss, lr, epoch
        eval_steps=100,  # for compute metrics
        report_to="tensorboard",
        save_total_limit=3,
        load_best_model_at_end=True,
        resume_from_checkpoint=resume_from_checkpoint
        if resume_from_checkpoint
        else None,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
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
        label_ids, predictions, labels=[1, 2, 3, 4, 5, 6], display_labels=LABELS
    )

    model_desc = f"bs{train_batch_size}_{model_save_name}_{input}{freeze_desc}"
    save_conf_matrix(disp=disp, model_name=model_desc)


if __name__ == "__main__":
    main()
