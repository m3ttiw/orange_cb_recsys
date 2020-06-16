from abc import abstractmethod

from orange_cb_recsys.evaluation.metrics import Metric
from orange_cb_recsys.utils.const import logger

import pandas as pd


class ClassificationMetric(Metric):
    """
    Abstract class that generalize classification metrics.
    A classification metric measure if
    known relevant items are predicted as relevant

    Args:
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
    """
    def __init__(self, relevant_threshold):
        self.__relevant_threshold = relevant_threshold

    def _get_labels(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        relevant_rank = truth[truth['rating'] >= self.__relevant_threshold]
        content_truth = pd.Series(relevant_rank['to_id'].values)
        content_prediction = pd.Series(predictions['to_id'].values)
        content_prediction = content_prediction[:content_truth.size]

        return content_truth, content_prediction

    @abstractmethod
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Method that execute the classification metric computation

        Args:
              truth (pd.DataFrame): dataframe whose columns are: to_id, rating
              predictions (pd.DataFrame): dataframe whose columns are: to_id, rating
        """
        raise NotImplementedError


class Precision(ClassificationMetric):
    """
    Precision

    Args:
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
    """
    def __init__(self, relevant_threshold):
        super().__init__(relevant_threshold)

    def __str__(self):
        return "Precision"

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
        """
        Compute the recall of the given ranking (predictions)
        based on the truth ranking

        Args:
              truth (pd.DataFrame): dataframe whose columns are: to_id, rating
              predictions (pd.DataFrame): dataframe whose columns are: to_id, rating

        Returns:
            (float): precision
        """
        logger.info("Computing precision")
        prediction_labels, truth_labels = super()._get_labels(predictions, truth)
        return prediction_labels.isin(truth_labels).sum() / len(prediction_labels)


class Recall(ClassificationMetric):
    """
    Recall

    Args:
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
    """
    def __init__(self, relevant_threshold):
        super().__init__(relevant_threshold)

    def __str__(self):
        return "Recall"

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
        """
        Compute the recall of the given ranking (predictions)
        based on the truth ranking

        Args:
              truth (pd.DataFrame): dataframe whose columns are: to_id, rating
              predictions (pd.DataFrame): dataframe whose columns are: to_id, rating

        Returns:
            (float): recall
        """
        logger.info("Computing recall")
        prediction_labels, truth_labels = super()._get_labels(predictions, truth)
        return prediction_labels.isin(truth_labels).sum() / len(truth_labels)


class MRR(ClassificationMetric):
    """
    MRR

    Args:
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
    """
    def __init__(self, relevant_threshold):
        super().__init__(relevant_threshold)

    def __str__(self):
        return "MRR"

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
        """
        Compute the Mean Reciprocal Rank metric

        Args:
              truth (pd.DataFrame): dataframe whose columns are: to_id, rating
              predictions (pd.DataFrame): dataframe whose columns are: to_id, rating

        Returns:
            (float): the mrr value
        """
        logger.info("Computing MRR")

        prediction_labels, truth_labels = super()._get_labels(predictions, truth)

        mrr = 0
        n = len(truth_labels)
        if n == 0:
            return 0
        for t_index, t_value in truth_labels.iteritems():
            for p_index, p_value in prediction_labels.iteritems():
                if t_value == p_value:
                    mrr += (int(t_index) + 1) / (int(p_index) + 1)
        return mrr / len(truth_labels)


class FNMeasure(ClassificationMetric):
    """
    FnMeasure

    Args:
        n (int): multiplier
        relevant_threshold: specify the minimum value to consider
            a truth frame row as relevant
    """
    def __init__(self, n, relevant_threshold: float):
        super().__init__(relevant_threshold)
        self.__n = n

    def __str__(self):
        return "F" + str(self.__n)

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame) -> float:
        """
        Compute the Fn measure of the given ranking (predictions)
        based on the truth ranking

        Args:
              truth (pd.DataFrame): dataframe whose columns are: to_id, rating
              predictions (pd.DataFrame): dataframe whose columns are: to_id, rating

        Returns:
            score (float): Fn value
        """

        logger.info("Computing FN")

        prediction_labels, truth_labels = super()._get_labels(predictions, truth)
        precision = prediction_labels.isin(truth_labels).sum() / len(prediction_labels)
        recall = prediction_labels.isin(truth_labels).sum() / len(truth_labels)

        return (1 + (self.__n ** 2)) * ((precision * recall) / ((self.__n ** 2) * precision + recall))
