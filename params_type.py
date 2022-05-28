from enum import Enum


class ModelType(str, Enum):
    distil_roberta = "distilroberta-base"
    distil_bert = "distilbert-base-uncased"


class TaskType(str, Enum):
    classification = "classification"
    ordinal_regression = "ordinal_regression"
    ordinal_regression_coral = "ordinal_regression_coral"
