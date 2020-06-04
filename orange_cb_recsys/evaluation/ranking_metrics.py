from typing import List
import pandas as pd
import numpy as np


def perform_precision(prediction_labels: pd.Series, truth_labels: pd.Series) -> float:
    """
    Returns the precision of the given ranking (predictions)
    based on the truth ranking
    Args:
        prediction_labels:
        truth_labels:

    Returns:

    """
    return prediction_labels.isin(truth_labels).sum() / len(prediction_labels)


def perform_recall(prediction_labels: pd.Series, truth_labels: pd.Series) -> float:
    """
    Compute the recall of the given ranking (predictions)
    based on the truth ranking
    Args:
        prediction_labels:
        truth_labels:

    Returns:

    """
    return prediction_labels.isin(truth_labels).sum() / len(truth_labels)


def perform_Fn(precision: float, recall: float, n: int = 1) -> float:
    """
    Compute the Fn measure of the given ranking (predictions)
    based on the truth ranking
    Args:
        precision:
        recall:
        n:

    Returns:

    """
    return (1 + (n ** 2)) * ((precision * recall) / ((n ** 2) * precision + recall))


def perform_DCG(gain_values: pd.Series) -> List[float]:
    """
    Compute the DCG array of a gain vector
    Args:
        gain_values:

    Returns:

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
    Compute the NDCG measure using Truth rank as ideal DCG
    Args:
        predictions:
        truth:

    Returns:

    """
    idcg = perform_DCG(pd.Series(truth['rating'].values))

    col = ["item", "rating"]
    new_predicted = pd.DataFrame(columns=col)
    for index, predicted_row in predictions.iterrows():
        predicted_item = predicted_row['item']
        truth_row = truth.loc[truth['item'] == predicted_item]
        truth_score = truth_row['rating'].values[0]
        new_predicted = new_predicted.append({'item': predicted_item, 'rating': truth_score}, ignore_index=True)

    dcg = perform_DCG(gain_values=pd.Series(new_predicted['rating'].values))
    ndcg = []
    for i, ideal in enumerate(idcg):
        try:
            ndcg.append(dcg[i] / ideal)
        except IndexError:
            break
    return ndcg


def perform_MRR(predictions_labels: pd.Series, truth_labels: pd.Series) -> float:
    """

    Args:
        predictions_labels:
        truth_labels:

    Returns:

    """
    mrr = 0
    for t_index, t_value in truth_labels.iteritems():
        for p_index, p_value in predictions_labels.iteritems():
            if t_value == p_value:
                mrr += (t_index + 1) / (t_value + 1)
    return mrr / len(truth_labels)
