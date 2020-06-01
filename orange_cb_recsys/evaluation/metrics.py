from typing import Dict
import pandas as pd
import numpy as np


def perform_precision(predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
    """
    Calculates the precision of the recommendations provided

    Args:
        predictions (pd.DataFrame): Series containing the predicted ratings
        truth (pd.DataFrame): Series containing the truth values

    Returns:
        float: precision
    """


def perform_recall(predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
    """
    Calculates the recall of the recommendations provided

    Args:
        predictions (pd.DataFrame): Series containing the predicted ratings
        truth (pd.DataFrame): Series containing the truth values

    Returns:
        float: recall
    """


def perform_f1(precision, recall) -> float:
    """
    Calculates the f1-measure of the recommendations provided

    Args:
        precision (float): Precision of the recommendations
        recall (float): Recall of the recommendations

    Returns:
        float: f1 measure
    """
    return 2 * ((precision * recall) / (precision + recall))


def perform_dcg(scores: pd.DataFrame):
    """
    Calculates the DCG of a given Series of scores

    Args:
        scores (pd.DataFrame): Series of scores of which the function will find the DCG

    Returns:
        dcg (float): value of the DCG
    """


def perform_ndcg(predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
    """
    Calculates the NDCG, given by the ratio between the DCG of the recommendations provided and the
    Ideal-DCG, represented by the DCG of the truth base

    Args:
        predictions (pd.DataFrame): Series containing the predicted ratings
        truth (pd.DataFrame): Series containing the truth values

    Returns:
        float: value of the ndcg
    """


def perform_ranking_metrics(predictions: pd.DataFrame, truth: pd.DataFrame) -> Dict[str, object]:
    """
    Performs the metrics for evaluating the ranking phase and returns their values

    Args:
        predictions (pd.Series): Series containing the predicted ratings
        truth (pd.Series): Series containing the truth values

    Returns:
        results (Dict[str, object]): Python dictionary where the keys are the names of the metrics and the
        values are the corresponding values
    """
    precision = perform_precision(predictions, truth)
    recall = perform_recall(predictions, truth)
    results = {
        "precision": precision,
        "recall": recall,
        "F1": perform_f1(precision, recall),
        "NDCG": perform_ndcg(predictions, truth)
    }
    return results


def perform_gini_index():
    pass


def perform_pop_recs_correlation():
    pass


def perform_fairness_metrics() -> Dict[str, object]:

    results = {
        "precision": perform_gini_index(),
        "pop_recs_correlation": perform_pop_recs_correlation()
    }
    return results


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


def perform_prediction_metrics(predictions: pd.Series, truth: pd.Series) -> Dict[str, object]:
    """
    Performs the metrics for evaluating the rating prediction phase and returns their values

    Args:
        predictions (pd.Series): Series containing the predicted ratings
        truth (pd.Series): Series containing the truth rating values

    Returns:
        results (Dict[str, object]): Python dictionary where the keys are the names of the metrics and the
        values are the corresponding values
    """
    results = {
        "RMSE": perform_rmse(predictions, truth),
        "MAE": perform_mae(predictions, truth)
    }
    return results
