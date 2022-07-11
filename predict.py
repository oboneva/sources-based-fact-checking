from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from data_loading_utils import load_datasplits_urls, load_splitted_train_split
from fc_dataset import FCDataset
from metrics_constants import LABELS
from model_params_utils import decode_params_from_output_dir
from ordinal_prediction_utils import prediction2label
from params_type import ModelType, TaskType


def get_split_urls_name(has_hold_out_train: bool):
    urls_test, urls_val, urls_train = load_datasplits_urls(
        urls_path="data/urls_split_stratified.json"
    )

    urls_train_less, urls_train_more = load_splitted_train_split(
        urls_path="data/urls_train_split_90_10.json", ratio=0.1
    )

    if has_hold_out_train:
        return zip([urls_test, urls_train_less], ["test", "train10"])
    else:
        return zip([urls_test, urls_val, urls_train], ["test", "val", "train"])


def get_prediction_setup(checkpoint: str, reverse_labels: bool, model_type: ModelType):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = LABELS

    if reverse_labels:
        labels = labels[::-1]

    label2id = {labels[i]: i for i in range(len(labels))}
    id2label = {id: label for label, id in label2id.items()}
    num_labels = len(labels)

    tokenizer = AutoTokenizer.from_pretrained(model_type.value)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(model=model, data_collator=data_collator)

    return trainer, label2id, tokenizer, device


def get_predictions_for_datasplit(
    reverse_labels: bool, task_type: TaskType, trainer, datasplit
):
    predictions, label_ids, _ = trainer.predict(datasplit)
    label_ids = np.argmax(label_ids, axis=-1)

    if task_type is TaskType.ordinal_regression:
        raw_predictions = torch.sigmoid(torch.tensor(predictions))
        predictions = prediction2label(raw_predictions)
    elif task_type is TaskType.classification:
        raw_predictions = torch.tensor(predictions)
        predictions = torch.tensor(np.argmax(predictions, axis=-1))
    else:
        raise ValueError("Unsupported task")

    if reverse_labels:
        predictions = torch.sub(torch.full(predictions.size(), 5), predictions)
        label_ids = torch.tensor(label_ids)
        label_ids = torch.sub(torch.full(label_ids.size(), 5), label_ids).numpy()

    return predictions, raw_predictions, label_ids


def get_test_predictions(model_checkpoint: str):
    (
        reverse_labels,
        encode_author,
        encoded_input,
        model_type,
        task_type,
    ) = decode_params_from_output_dir(output_dir=model_checkpoint)

    trainer, label2id, tokenizer, device = get_prediction_setup(
        model_checkpoint, reverse_labels, model_type
    )

    urls_test, _, _ = load_datasplits_urls(urls_path="data/urls_split_stratified.json")
    aticles_dir = "data/articles_parsed_clean_date"

    test_dataset = FCDataset(
        urls=urls_test,
        articles_dir=aticles_dir,
        encoded_input=encoded_input,
        encode_author=encode_author,
        all_labels2id=label2id,
        tokenizer=tokenizer,
        device=device,
    )

    predictions = get_predictions_for_datasplit(
        reverse_labels=reverse_labels,
        task_type=task_type,
        trainer=trainer,
        datasplit=test_dataset,
    )

    return predictions


def save_predictions(checkpoint: str, has_hold_out_train: bool):
    dir_parts = checkpoint.split("/")
    save_dir = "/".join(dir_parts[:-1])
    save_dir = save_dir + "/predictions"

    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    (
        reverse_labels,
        encode_author,
        encoded_input,
        model_name,
        task_type,
    ) = decode_params_from_output_dir(output_dir=checkpoint)

    trainer, label2id, tokenizer, device = get_prediction_setup(
        checkpoint, reverse_labels, model_name
    )

    urls_name = get_split_urls_name(has_hold_out_train=has_hold_out_train)
    articles_dir = "data/articles_parsed_clean_date"

    for urls, set_name in urls_name:
        dataset = FCDataset(
            urls=urls,
            articles_dir=articles_dir,
            encoded_input=encoded_input,
            encode_author=encode_author,
            all_labels2id=label2id,
            tokenizer=tokenizer,
            device=device,
        )

        preds, raw, y_true = get_predictions_for_datasplit(
            reverse_labels=reverse_labels,
            task_type=task_type,
            trainer=trainer,
            datasplit=dataset,
        )

        np.save(f"{save_dir}/{set_name}_preds.npy", preds)
        np.save(f"{save_dir}/{set_name}_raw.npy", raw)
        np.save(f"{save_dir}/{set_name}_y_true.npy", y_true)


def main():
    # save_predictions(checkpoint="", has_hold_out_train=True)
    pass


if __name__ == "__main__":
    main()
