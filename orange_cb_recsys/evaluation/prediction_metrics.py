import pandas as pd
import numpy as np


def perform_rmse(predictions: pd.Series, truth: pd.Series) -> float:
    """
    Compute the RMSE

    Args:
        predictions (pd.Series): Series containing the predicted ratings
        truth (pd.Series): Series containing the truth rating values

    Returns:
        float: The Root Mean Squared Error
    """
    assert len(predictions) == len(truth), "The predictions series and the truth series must have the same size"
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
    assert len(predictions) == len(truth), "The predictions series and the truth series must have the same size"
    abs_diff = (predictions - truth).apply(abs)
    return np.mean(abs_diff)