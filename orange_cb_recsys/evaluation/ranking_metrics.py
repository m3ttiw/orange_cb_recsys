from typing import List
import pandas as pd
import numpy as np


def perform_precision(prediction_labels: pd.Series, truth_labels: pd.Series) -> float:
    """
    Returns the precision of the given ranking (predictions)
    based on the truth ranking

    Args:
        prediction_labels (pd.Series): pandas Series wich contains predicted "labels"
        truth_labels (pd.Series): pandas Series wich contains truth "labels"

    Returns:
        score (float): precision
    """
    return prediction_labels.isin(truth_labels).sum() / len(prediction_labels)


def perform_recall(prediction_labels: pd.Series, truth_labels: pd.Series) -> float:
    """
    Compute the recall of the given ranking (predictions)
    based on the truth ranking

    Args:
        prediction_labels (pd.Series): pandas Series wich contains predicted "labels"
        truth_labels (pd.Series): pandas Series wich contains truth "labels"

    Returns:
        score (float): recall
    """
    return prediction_labels.isin(truth_labels).sum() / len(truth_labels)


def perform_Fn(precision: float, recall: float, n: int = 1) -> float:
    """
    Compute the Fn measure of the given ranking (predictions)
    based on the truth ranking

    Args:
        precision (float): precision of the rank
        recall (float): recall of the rank
        n (int): multiplier

    Returns:
        score (float): Fn value
    """
    return (1 + (n ** 2)) * ((precision * recall) / ((n ** 2) * precision + recall))


def perform_DCG(gain_values: pd.Series) -> List[float]:
    """
    Compute the Discounted Cumulative Gain array of a gain vector
    Args:
        gain_values (pd.Series): array of gains

    Returns:
        dcg (List[float]): array of dcg
    """
    dcg = []
    for i, gain in enumerate(gain_values):
        if i == 0:
            dcg.append(gain)
        else:
            dcg.append((gain / np.log2(i + 1)) + dcg[i - 1])
    return dcg


def perform_NDCG(predictions: pd.DataFrame, truth: pd.DataFrame) -> List[float]:
    """
    Compute the Normalized DCG measure using Truth rank as ideal DCG
    Args:
        predictions (pd.DataFrame): each row contains index(the rank position), label, value predicted
        truth (pd.DataFrame): the real rank each row contains index(the rank position), label, value

    Returns:
        ndcg (List[float]): array of ndcg
    """
    idcg = perform_DCG(pd.Series(truth['rating'].values))

    col = ["to_id", "rating"]
    new_predicted = pd.DataFrame(columns=col)
    for index, predicted_row in predictions.iterrows():
        predicted_item = predicted_row['to_id']
        truth_row = truth.loc[truth['to_id'] == predicted_item]
        truth_score = truth_row['rating'].values[0]
        new_predicted = new_predicted.append({'to_id': predicted_item, 'rating': truth_score}, ignore_index=True)

    dcg = perform_DCG(gain_values=pd.Series(new_predicted['rating'].values))
    ndcg = []
    for i, ideal in enumerate(idcg):
        try:
            ndcg.append(dcg[i] / ideal)
        except IndexError:
            break
    return ndcg


def perform_MRR(prediction_labels: pd.Series, truth_labels: pd.Series) -> float:
    """
    Compute the Mean Reciprocal Rank metric

    Args:
        prediction_labels (pd.Series): pandas Series wich contains predicted "labels"
        truth_labels (pd.Series): pandas Series wich contains truth "labels"

    Returns:
        score (float): the mrr value
    """
    mrr = 0
    for t_index, t_value in truth_labels.iteritems():
        for p_index, p_value in prediction_labels.iteritems():
            if t_value == p_value:
                mrr += (int(t_index) + 1) / (int(p_index) + 1)
    return mrr / len(truth_labels)


def perform_correlation(prediction_labels: pd.Series, truth_labels: pd.Series, method='pearson') -> float:
    """
    Compute the correlation between the two ranks of labels

    Args:
        prediction_labels (pd.Series): pandas Series wich contains predicted "labels"
        truth_labels (pd.Series): pandas Series wich contains truth "labels"
        method: {'pearson, 'kendall', 'spearman'} or callable

    Returns:
        score (float): value of the specified correlation metric
    """
    t_series = pd.Series()
    p_series = pd.Series()
    for t_index, t_value in truth_labels.iteritems():
        t_series = t_series.append(pd.Series(int(t_index)))
        for p_index, p_value in prediction_labels.iteritems():
            if t_value == p_value:
                p_series = p_series.append(pd.Series(int(p_index)))

    return t_series.corr(other=p_series, method=method)



