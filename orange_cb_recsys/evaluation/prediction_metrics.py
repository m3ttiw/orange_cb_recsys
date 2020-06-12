import pandas as pd
import numpy as np

from orange_cb_recsys.utils.const import logger


def perform_rmse(predictions: pd.Series, truth: pd.Series) -> float:
    """
    Compute the RMSE metric

    Args:
        predictions (pd.Series): Series containing the predicted ratings
        truth (pd.Series): Series containing the truth rating values

    Returns:
        (float): The Root Mean Squared Error
    """
    logger.info("Computing RMSE")

    if len(predictions) != len(truth):
        raise Exception
    diff = predictions - truth
    sq = np.square(diff)
    return np.sqrt(np.mean(sq))


def perform_mae(predictions: pd.Series, truth: pd.Series) -> float:
    """
    Compute the MAE metric

    Args:
        predictions (pd.Series): Series containing the predicted ratings
        truth (pd.Series): Series containing the truth rating values

    Returns:
        (float): The Mean Average Error
    """
    logger.info("Computing MAE")

    if len(predictions) != len(truth):
        raise Exception
    abs_diff = (predictions - truth).apply(abs)
    return np.mean(abs_diff)
