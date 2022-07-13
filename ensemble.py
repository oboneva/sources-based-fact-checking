import json
from typing import List

import numpy as np
import torch
from mord import LogisticAT
from numpy import load
from sklearn.metrics import ConfusionMatrixDisplay

from compute_metrics import compute_metrics
from data_loading_utils import load_datasplits_urls, load_splitted_train_split
from metrics_constants import LABELS
from predict import get_test_predictions
from results_utils import save_conf_matrix


def load_data_from_urls(articles_dir: str, urls):
    infos = []
    for url in urls:
        article_filename = url.split("/")[-2]

        with open(f"{articles_dir}/{article_filename}.json") as f:
            data = json.load(f)

        infos.append(data["stats"])

    infos = np.array(infos)

    return infos


def load_stats_data(urls_path: str, articles_dir: str, blending: bool):
    urls_test, _, urls_train = load_datasplits_urls(urls_path=urls_path)
    if blending:
        urls_train_less, _ = load_splitted_train_split(
            urls_path="./data/urls_train_split_90_10.json", ratio=0.1
        )
        urls_train = urls_train_less

    test_data = load_data_from_urls(articles_dir=articles_dir, urls=urls_test)
    train_data = load_data_from_urls(articles_dir=articles_dir, urls=urls_train)

    return test_data, train_data


def avg(predictions):
    predictions = torch.add(predictions[0], predictions[1])
    for i in range(2, len(predictions)):
        predictions = torch.add(predictions, predictions[i])

    predictions = torch.div(predictions, len(predictions))
    predictions = predictions.numpy()

    predictions = [round(pred) for pred in predictions]

    return predictions


def avg_ensemble():
    predictions1, _, label_ids = get_test_predictions(model_checkpoint="")
    predictions2, _, _ = get_test_predictions(model_checkpoint="")
    predictions3, _, _ = get_test_predictions(model_checkpoint="")

    predictions = avg(predictions=[predictions1, predictions2, predictions3])

    print(compute_metrics(label_ids, predictions))

    disp = ConfusionMatrixDisplay.from_predictions(
        label_ids, predictions, labels=[0, 1, 2, 3, 4, 5], display_labels=LABELS
    )

    save_conf_matrix(disp=disp, model_name="")


def train_eval_meta_model(
    models_dirs: List[str],
    meta_model,
    is_blending: bool,
    train_nli,
    test_nli,
    add_nli: bool,
    train_stance,
    test_stance,
    add_stance: bool,
    is_reversed_stance: bool,
):
    train = "train10" if is_blending else "train"
    test = "test"

    y_true_train = load(f"{models_dirs[0]}/predictions/{train}_y_true.npy")
    y_true_test = load(f"{models_dirs[0]}/predictions/{test}_y_true.npy")

    preds_type = "raw"

    concatenated = load(f"{models_dirs[0]}/predictions/{train}_{preds_type}.npy")
    concatenated_test = load(f"{models_dirs[0]}/predictions/{test}_{preds_type}.npy")

    for i in range(1, len(models_dirs)):
        raw_train = load(f"{models_dirs[i]}/predictions/{train}_{preds_type}.npy")
        raw = load(f"{models_dirs[i]}/predictions/{test}_{preds_type}.npy")

        concatenated = np.concatenate((concatenated, raw_train), axis=1)
        concatenated_test = np.concatenate((concatenated_test, raw), axis=1)

    if add_nli:
        concatenated = np.concatenate((concatenated, train_nli), axis=1)
        concatenated_test = np.concatenate((concatenated_test, test_nli), axis=1)

    if add_stance:
        concatenated = np.concatenate((concatenated, train_stance), axis=1)
        concatenated_test = np.concatenate((concatenated_test, test_stance), axis=1)

    meta_model.fit(concatenated, y_true_train)
    y_pred = meta_model.predict(concatenated_test)

    y_pred = [round(pred) for pred in y_pred]

    metrics = compute_metrics(y_true=y_true_test, y_pred=y_pred)

    # formater = "{:.3f} {:.3f} {:.3f} {:.4f} {:.4f}"
    # formatted_string = formater.format(
    #     metrics["accuracy"] * 100,
    #     metrics["f1"] * 100,
    #     metrics["recall"] * 100,
    #     metrics["mae"],
    #     metrics["mse"],
    # ).replace(".", ",")
    # print(metrics)
    # print(formatted_string)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true_test, y_pred, labels=[0, 1, 2, 3, 4, 5], display_labels=LABELS
    )

    nli_desc = "+nli" if add_nli else ""
    stance_desc = (
        ""
        if not add_stance
        else ("+stance" if not is_reversed_stance else "+rev_stance")
    )
    model_name = (
        f"{type(meta_model).__name__}_{len(models_dirs)}best{nli_desc}{stance_desc}"
    )
    save_conf_matrix(disp=disp, model_name=model_name)

    return metrics["mae"]


def experiments_stacking():
    is_blending = True

    add_nli = False
    add_stance = False
    is_reversed_stance = False

    articles_dir_nli = "./data/articles_nli"
    articles_dir_stance = (
        "./data/articles_stance_reversed"
        if is_reversed_stance
        else "./data/articles_stance"
    )

    test_data, train_data = load_stats_data(
        articles_dir=articles_dir_nli,
        urls_path="./data/urls_split_stratified.json",
        blending=is_blending,
    )

    test_data_stance, train_data_stance = load_stats_data(
        articles_dir=articles_dir_stance,
        urls_path="./data/urls_split_stratified.json",
        blending=is_blending,
    )

    model_params = {
        "max_iter": 25000,
    }
    print(model_params)

    model = LogisticAT(**model_params)

    checkpoints = []

    _ = train_eval_meta_model(
        models_dirs=checkpoints,
        meta_model=model,
        is_blending=is_blending,
        train_nli=train_data,
        test_nli=test_data,
        add_nli=add_nli,
        train_stance=train_data_stance,
        test_stance=test_data_stance,
        add_stance=add_stance,
        is_reversed_stance=is_reversed_stance,
    )


def main():
    # experiments_stacking()
    pass


if __name__ == "__main__":
    main()
