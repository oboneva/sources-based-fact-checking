from enum import Enum


class ModelType(str, Enum):
    roberta_base = "roberta-base"
    distil_roberta = "distilroberta-base"
    distil_bert = "distilbert-base-uncased"

    @classmethod
    def from_str(cls, input_str):
        for member in cls:
            if member.value == input_str:
                return member
        raise ValueError(f"{cls.__name__} has no value matching {input_str}")


class TaskType(str, Enum):
    classification = "classification"
    ordinal_regression = "ordinal_regression"
    ordinal_regression_coral = "ordinal_regression_coral"
