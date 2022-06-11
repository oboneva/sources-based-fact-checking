import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from data_loading_utils import load_datasplits_urls, load_splitted_train_split
from fc_dataset import EncodedInput, FCDataset
from metrics_constants import LABELS
from params_type import ModelType


def prediction2label(pred):
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1


def save_predictions(
    checkpoint: str,
    save_dir: str,
    reverse_labels: bool,
    ordinal: bool,
    encoded_input: EncodedInput,
    encode_author: bool,
    has_hold_out_train: bool,
):
    predictions = get_predictions(
        reverse_labels=reverse_labels,
        ordinal=ordinal,
        encoded_input=encoded_input,
        encode_author=encode_author,
        model_checkpoint=checkpoint,
        has_hold_out_train=has_hold_out_train,
    )

    if has_hold_out_train:
        preds, raw, y_true, preds_train, raw_train, y_true_train = predictions
        np.save(f"{save_dir}/preds.npy", preds)
        np.save(f"{save_dir}/raw.npy", raw)
        np.save(f"{save_dir}/y_true.npy", y_true)
        np.save(f"{save_dir}/preds_train.npy", preds_train)
        np.save(f"{save_dir}/raw_train.npy", raw_train)
        np.save(f"{save_dir}/y_true_train.npy", y_true_train)
    else:
        preds, raw, y_true = predictions
        np.save(f"{save_dir}/preds.npy", preds)
        np.save(f"{save_dir}/raw.npy", raw)
        np.save(f"{save_dir}/y_true.npy", y_true)


def get_predictions_for_datasplit(
    reverse_labels: bool, ordinal: bool, trainer, datasplit
):
    predictions, label_ids, _ = trainer.predict(datasplit)
    label_ids = np.argmax(label_ids, axis=-1)

    if ordinal:
        raw_predictions = torch.sigmoid(torch.tensor(predictions))
        predictions = prediction2label(raw_predictions)
    else:
        predictions = torch.tensor(np.argmax(predictions, axis=-1))

    if reverse_labels:
        predictions = torch.sub(torch.full(predictions.size(), 5), predictions)
        label_ids = torch.tensor(label_ids)
        label_ids = torch.sub(torch.full(label_ids.size(), 5), label_ids).numpy()

    return predictions, raw_predictions, label_ids


def get_predictions(
    reverse_labels: bool,
    ordinal: bool,
    encoded_input: EncodedInput,
    encode_author: bool,
    model_checkpoint: str,
    has_hold_out_train: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = LABELS

    if reverse_labels:
        labels = labels[::-1]

    label2id = {labels[i]: i for i in range(len(labels))}
    id2label = {id: label for label, id in label2id.items()}
    num_labels = len(labels)

    # 1. Prepare the Data.
    urls_test, _, _ = load_datasplits_urls(urls_path="data/urls_split_stratified.json")
    aticles_dir = "data/articles_parsed_clean_date"

    tokenizer = AutoTokenizer.from_pretrained(ModelType.distil_roberta.value)
    test_dataset = FCDataset(
        urls=urls_test,
        articles_dir=aticles_dir,
        encoded_input=encoded_input,
        encode_author=encode_author,
        all_labels2id=label2id,
        tokenizer=tokenizer,
        device=device,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(model=model, data_collator=data_collator)

    predictions = get_predictions_for_datasplit(
        reverse_labels=reverse_labels,
        ordinal=ordinal,
        trainer=trainer,
        datasplit=test_dataset,
    )

    if has_hold_out_train:
        urls_train_less, urls_train_more = load_splitted_train_split(
            urls_path="data/urls_train_split_90_10.json", ratio=0.1
        )

        hold_out_dataset = FCDataset(
            urls=urls_train_less,
            articles_dir=aticles_dir,
            encoded_input=encoded_input,
            encode_author=encode_author,
            all_labels2id=label2id,
            tokenizer=tokenizer,
            device=device,
        )

        predictions_train = get_predictions_for_datasplit(
            reverse_labels=reverse_labels,
            ordinal=ordinal,
            trainer=trainer,
            datasplit=hold_out_dataset,
        )
        predictions = predictions + predictions_train

    return predictions


def main():
    # save_predictions(
    #     checkpoint="",
    #     save_dir="",
    #     reverse_labels=True,
    #     ordinal=True,
    #     encoded_input=EncodedInput.TEXT,
    #     encode_author=False,
    #     has_hold_out_train=True,
    # )
    pass


if __name__ == "__main__":
    main()
