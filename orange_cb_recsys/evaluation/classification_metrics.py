from orange_cb_recsys.evaluation.metrics import Metric
from orange_cb_recsys.utils.const import logger

import pandas as pd


class ClassificationMetric(Metric):
    def __init__(self, relevance_threshold):
        self.__relevance_threshold = relevance_threshold

    def get_labels(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        relevant_rank = truth[truth['rating'] >= self.__relevance_threshold]
        content_truth = pd.Series(relevant_rank['to_id'].values)
        content_prediction = pd.Series(predictions['to_id'].values)
        content_prediction = content_prediction[:content_truth.size]

        return content_truth, content_prediction

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        raise NotImplementedError


class Precision(ClassificationMetric):
    def __init__(self, relevant_threshold):
        super().__init__(relevant_threshold)

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        logger.info("Computing precision")
        prediction_labels, truth_labels = super().get_labels(predictions, truth)
        return prediction_labels.isin(truth_labels).sum() / len(prediction_labels)


class Recall(ClassificationMetric):
    def __init__(self, relevant_threshold):
        super().__init__(relevant_threshold)

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Compute the recall of the given ranking (predictions)
        based on the truth ranking

        Returns:
            (float): recall
        """
        logger.info("Computing recall")
        prediction_labels, truth_labels = super().get_labels(predictions, truth)
        return prediction_labels.isin(truth_labels).sum() / len(truth_labels)


class MRR(ClassificationMetric):
    def __init__(self, relevant_threshold):
        super().__init__(relevant_threshold)

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Compute the Mean Reciprocal Rank metric

        Returns:
            (float): the mrr value
        """
        logger.info("Computing MRR")

        prediction_labels, truth_labels = super().get_labels(predictions, truth)

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
    def __init__(self, n, relevant_threshold: float):
        """
        Args:
            n (int): multiplier
        """
        super().__init__(relevant_threshold)
        self.__n = n

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Compute the Fn measure of the given ranking (predictions)
        based on the truth ranking

        Returns:
            score (float): Fn value
        """

        logger.info("Computing FN")

        prediction_labels, truth_labels = super().get_labels(predictions, truth)
        precision = prediction_labels.isin(truth_labels).sum() / len(prediction_labels)
        recall = prediction_labels.isin(truth_labels).sum() / len(truth_labels)

        return (1 + (self.__n ** 2)) * ((precision * recall) / ((self.__n ** 2) * precision + recall))
