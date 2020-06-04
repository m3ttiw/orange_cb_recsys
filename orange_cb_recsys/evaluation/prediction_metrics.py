import pandas as pd
import numpy as np
from warnings import warn


def perform_rmse(predictions: pd.Series, truth: pd.Series) -> float:
    """
    Compute the RMSE

    Args:
        predictions (pd.Series): Series containing the predicted ratings
        truth (pd.Series): Series containing the truth rating values

    Returns:
        float: The Root Mean Squared Error
    """
    if len(predictions) == len(truth):
        warn("The predictions series and the truth series must have the same size")
        return 0.0
    diff = predictions - truth
    sq = np.square(diff)
    return np.sqrt(np.mean(sq))


def perform_mae(predictions: pd.Series, truth: pd.Series) -> float:
    """
    Compute the RMSE

    Args:
        predictions (pd.Series): Series containing the predicted ratings
        truth (pd.Series): Series containing the truth rating values

    Returns:
        float: The Mean Average Error
    """
    if len(predictions) == len(truth):
        warn("The predictions series and the truth series must have the same size")
        return 0.0
    abs_diff = (predictions - truth).apply(abs)
    return np.mean(abs_diff)