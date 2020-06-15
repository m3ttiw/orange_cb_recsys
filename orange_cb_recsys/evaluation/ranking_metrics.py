import statistics
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr

from orange_cb_recsys.utils.const import logger


def perform_precision(prediction_labels: pd.Series, truth_labels: pd.Series) -> float:
    """
    Returns the precision of the given ranking (predictions)
    based on the truth ranking

    Args:
        prediction_labels (pd.Series): pandas Series which contains predicted "labels"
        truth_labels (pd.Series): pandas Series which contains truth "labels"

    Returns:
        score (float): precision
    """
    logger.info("Computing precision")
    return prediction_labels.isin(truth_labels).sum() / len(prediction_labels)


def perform_recall(prediction_labels: pd.Series, truth_labels: pd.Series) -> float:
    """
    Compute the recall of the given ranking (predictions)
    based on the truth ranking

    Args:
        prediction_labels (pd.Series): pandas Series wich contains predicted "labels"
        truth_labels (pd.Series): pandas Series wich contains truth "labels"

    Returns:
        (float): recall
    """
    logger.info("Computing recall")

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
    logger.info("Computing FN")

    return (1 + (n ** 2)) * ((precision * recall) / ((n ** 2) * precision + recall))


def perform_DCG(gain_values: pd.Series) -> List[float]:
    """
    Compute the Discounted Cumulative Gain array of a gain vector
    Args:
        gain_values (pd.Series): Series of gains

    Returns:
        dcg (List<float>): array of dcg
    """
    if gain_values.size == 0:
        return []
    dcg = []
    for i, gain in enumerate(gain_values):
        if i == 0:
            dcg.append(gain)
        else:
            dcg.append((gain / np.log2(i + 1)) + dcg[i - 1])
    return dcg


def perform_NDCG(predictions: pd.DataFrame, truth: pd.DataFrame,
                 split: Dict[int, Tuple[float, float]] = None
                 # ) -> List[float]:
                 ) -> float:
    """
    Compute the Normalized DCG measure using Truth rank as ideal DCG
    Args:
        split:
        predictions (pd.DataFrame): each row contains index(the rank position), label, value predicted
        truth (pd.DataFrame): the real rank each row contains index(the rank position), label, value

    Returns:
        ndcg (List[float]): array of ndcg
    """
    logger.info("Computing NDCG")

    def discrete(score_: float):
        if split is not None and len(split.keys()) != 0:

            shift_class = 0
            while 0 + shift_class not in split.keys():
                shift_class += 1
            shift_class += 1  # no negative
            for class_ in split.keys():
                min_, max_ = split[class_]
                if min_ <= score_ <= max_:  # assumption
                    return class_ + shift_class

            # if score_ not in split ranges
            if score_ > 0.0:
                return max(split.keys())
            return min(split.keys())

        return score_ + 1  # no negative, shift to range(0,2) from range (-1, 1)

    gain = []
    for idx, row in predictions.iterrows():
        label = row['to_id']
        score = discrete(truth.rating[truth['to_id'] == label].values[0])
        gain.append(score)
    gain = np.array(gain)
    # gain = predictions['rating'].values

    igain = gain.copy()
    igain[::-1].sort()
    idcg = perform_DCG(pd.Series(igain))
    dcg = perform_DCG(pd.Series(gain))
    ndcg = [dcg[x]/(idcg[x]) for x in range(len(idcg))]
    if len(ndcg) == 0:
        return 0.0
    return statistics.mean(ndcg)

"""
label_intersection = set(predictions[['to_id']].values.flatten()).intersection(
        set(truth[['to_id']].values.flatten())) 
idcg = perform_DCG(pd.Series(truth['rating'].values))
col = ["to_id", "rating"]
new_predicted = pd.DataFrame(columns=col)
for label in predictions['rating'].values:
    truth_row = truth.loc[truth['to_id'] == label]
    truth_score = truth_row['rating'].values[0]
    new_predicted = new_predicted.append({'to_id': label, 'rating': truth_score}, ignore_index=True)

dcg = perform_DCG(gain_values=pd.Series(new_predicted['rating'].values))
ndcg = []
for i, ideal in enumerate(idcg):
    try:
        ndcg.append(dcg[i] / ideal)
    except IndexError:
        break
    except ZeroDivisionError:
        ndcg.append(0.0)
return ndcg
"""


def perform_MRR(prediction_labels: pd.Series, truth_labels: pd.Series) -> float:
    """
    Compute the Mean Reciprocal Rank metric

    Args:
        prediction_labels (pd.Series): pandas Series wich contains predicted "labels"
        truth_labels (pd.Series): pandas Series wich contains truth "labels"

    Returns:
        (float): the mrr value
    """
    logger.info("Computing MRR")

    mrr = 0
    n = len(truth_labels)
    if n == 0:
        return 0
    for t_index, t_value in truth_labels.iteritems():
        for p_index, p_value in prediction_labels.iteritems():
            if t_value == p_value:
                mrr += (int(t_index) + 1) / (int(p_index) + 1)
    return mrr / len(truth_labels)


def perform_correlation(prediction_labels: pd.Series, truth_labels: pd.Series, method='pearson') -> float:
    """
    Compute the correlation between the two ranks of labels

    Args:
        prediction_labels (pd.Series): pandas Series which contains predicted "labels"
        truth_labels (pd.Series): pandas Series which contains truth "labels"
        method: {'pearson, 'kendall', 'spearman'} or callable

    Returns:
        (float): value of the specified correlation metric
    """
    logger.info("Computing %s correlation" % method)
    t_series = pd.Series()
    p_series = pd.Series()
    for t_index, t_value in truth_labels.iteritems():
        for p_index, p_value in prediction_labels.iteritems():
            if t_value == p_value:
                t_series = t_series.append(pd.Series(int(t_index)))
                p_series = p_series.append(pd.Series(int(p_index)))

    coef, p = 0, 0
    if method == 'pearson':
        coef, p = pearsonr(t_series, p_series)
    if method == 'kendall':
        coef, p = kendalltau(t_series, p_series)
    if method == 'spearman':
        coef, p = spearmanr(t_series, p_series)

    return coef

