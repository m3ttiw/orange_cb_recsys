import pandas as pd
import numpy as np

from orange_cb_recsys.evaluation.metrics import Metric
from orange_cb_recsys.utils.const import logger


class PredictionMetric(Metric):
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        raise NotImplementedError


class RMSE(PredictionMetric):
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Compute the RMSE metric

        Args:
            predictions (pd.Series): Series containing the predicted ratings
            truth (pd.Series): Series containing the truth rating values

        Returns:
            (float): The Root Mean Squared Error
        """
        logger.info("Computing RMSE")

        predictions = pd.Series(predictions['rating'].values, name="rating", dtype=float)
        truth = pd.Series(truth['score'].values, name="rating", dtype=float)

        if len(predictions) != len(truth):
            raise Exception
        diff = predictions - truth
        sq = np.square(diff)
        return np.sqrt(np.mean(sq))


class MAE(PredictionMetric):
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Compute the MAE metric

        Args:
            predictions (pd.Series): Series containing the predicted ratings
            truth (pd.Series): Series containing the truth rating values

        Returns:
            (float): The Mean Average Error
        """
        logger.info("Computing MAE")

        predictions = pd.Series(predictions['rating'].values, name="rating", dtype=float)
        truth = pd.Series(truth['score'].values, name="rating", dtype=float)

        if len(predictions) != len(truth):
            raise Exception
        abs_diff = (predictions - truth).apply(abs)
        return np.mean(abs_diff)
