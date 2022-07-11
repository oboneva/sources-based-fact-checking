import json
from os import listdir
from os.path import isfile, join
from typing import Optional

from fc_dataset import EncodedInput
from params_type import ModelType, TaskType


def model_params_filename_for_dir(dir: str) -> Optional[str]:
    filenames = [f for f in listdir(dir) if isfile(join(dir, f))]

    for filename in filenames:
        if "model_params" in filename:
            return filename

    return None


def decode_params_from_output_dir(output_dir: str):
    model_params_filename = model_params_filename_for_dir(dir=output_dir)

    with open(f"{output_dir}/{model_params_filename}") as f:
        data = json.load(f)

    reverse_labels = data["reverse_labels"]
    model_name = ModelType.from_str(data["model_name"])
    encoded_input = EncodedInput[data["encoded_input"]]
    encode_author = data["encode_author"]
    task_type = TaskType[data["task_type"]]

    return reverse_labels, encode_author, encoded_input, model_name, task_type
