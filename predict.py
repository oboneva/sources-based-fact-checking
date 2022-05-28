import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from data_loading_utils import load_datasplits_urls
from fc_dataset import EncodedInput, FCDataset
from metrics_constants import LABELS
from params_type import ModelType


def prediction2label(pred):
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1


def get_predictions(reverse_labels: bool, ordinal: bool, model_checkpoint):
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
        encoded_input=EncodedInput.TEXT,
        encode_author=True,
        label2id=label2id,
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

    predictions, label_ids, _ = trainer.predict(test_dataset)
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

    return predictions, label_ids
