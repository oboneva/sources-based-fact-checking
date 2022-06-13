from enum import Enum


class ModelType(str, Enum):
    roberta_base = "roberta-base"
    distil_roberta = "distilroberta-base"
    distil_bert = "distilbert-base-uncased"


class TaskType(str, Enum):
    classification = "classification"
    ordinal_regression = "ordinal_regression"
    ordinal_regression_coral = "ordinal_regression_coral"
