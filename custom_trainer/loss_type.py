from enum import Enum


class Loss(str, Enum):
    CEL = "CEL"
    WCEL = "WCEL"
    MAE = "MAE"
    MSE = "MSE"
    WMSE = "WMSE"
    FL = "FL"
