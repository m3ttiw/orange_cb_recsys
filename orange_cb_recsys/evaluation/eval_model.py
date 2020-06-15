import os
from typing import List, Dict

import pandas as pd

from orange_cb_recsys.evaluation.classification_metrics import ClassificationMetric
from orange_cb_recsys.evaluation.fairness_metrics import FairnessMetric
from orange_cb_recsys.evaluation.metrics import Metric
from orange_cb_recsys.evaluation.partitioning import Partitioning
from orange_cb_recsys.evaluation.prediction_metrics import PredictionMetric
from orange_cb_recsys.evaluation.ranking_metrics import RankingMetric
from orange_cb_recsys.recsys.algorithm import ScorePredictionAlgorithm
from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.recsys.recsys import RecSys
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.load_content import remove_not_existent_items


class RankingMetricsConfig:
    def __init__(self, relevant_threshold: float, relevance_split):
        self.__relevant_threshold = relevant_threshold
        self.__relevance_split = relevance_split

    def get_relevant_threshold(self):
        return self.__relevant_threshold

    def get_relevance_split(self):
        return self.__relevance_split


class EvalModel:
    """
    Class for the evaluation
    Args:
        config (RecSysConfig): Configuration of the recommender system that will be internally
            created
        partitioning (Partitioning): Partitioning technique
    """
    def __init__(self, config: RecSysConfig,
                 partitioning: Partitioning,
                 prediction_metrics: bool,
                 ranking_metrics: bool,
                 fairness_metrics: bool,
                 serendipity_novelty_metrics:  bool,
                 metric_list: List[Metric] = None):
        if metric_list is None:
            metric_list = []
        self.__prediction_metrics = prediction_metrics
        self.__ranking_metrics = ranking_metrics
        self.__fairness_metrics = fairness_metrics
        self.__serendipity_novelty_metrics = serendipity_novelty_metrics
        self.__metric_list = metric_list
        self.__config: RecSysConfig = config
        self.__partitioning = partitioning

    def get_fairness_metric_list(self):
        for metric in self.__fairness_metrics:
            if isinstance(metric, FairnessMetric):
                yield metric

    def get_prediction_metric_list(self):
        for metric in self.__metric_list:
            if isinstance(metric, PredictionMetric):
                yield metric

    def get_ranking_metric_list(self):
        for metric in self.__metric_list:
            if isinstance(metric, RankingMetric) or \
                    isinstance(metric, ClassificationMetric):
                yield metric

    def append_metric(self, metric: Metric):
        self.__metric_list.append(metric)

    def get_metrics(self):
        for metric in self.__metric_list:
            yield metric

    def fit(self):
        """
        This method performs the evaluation by initializing internally a recommender system that produces
            recommendations for all the users in the directory specified in the configuration phase.
            The evaluation is performed by creating a training set, and a test set with its corresponding
            truth base.

        Returns:
            Tuple<prediction_metric_results, ranking_metric_results, fairness_metrics_results>: Three
                different DataFrames. Each DataFrame has a 'from' column, representing the user_ids for
                which the recommendations are provided, and then one different column for every metric
                performed. The returned DataFrames contain one row per user, and the corresponding
                metric values are given by the mean of the values obtained for that user.
        """
        # initialize recommender to call for prediction computing
        recsys = RecSys(self.__config)

        # get all users in specified directory
        logger.info("Loading user instances")
        user_id_list = [os.path.splitext(filename)[0] for filename in os.listdir(self.__config.get_users_directory())]

        # define results structure
        prediction_metric_results = pd.DataFrame()
        ranking_metric_results = pd.DataFrame()

        # calculate prediction metrics
        if self.__prediction_metrics:
            if self.__config.get_score_prediction_algorithm() is None:
                raise ValueError("You must set score prediction algorithm to compute prediction metrics")

            for user_id in user_id_list:
                logger.info("User %s" % user_id)
                logger.info("Loading user ratings")

                user_ratings = self.__config.get_rating_frame()[
                    self.__config.get_rating_frame()['from_id'] == user_id]
                user_ratings = user_ratings.sort_values(['to_id'], ascending=True)

                try:
                    self.__partitioning.set_dataframe(user_ratings)
                except ValueError:
                    continue

                for partition_index in self.__partitioning:
                    result_dict = {}
                    logger.info("Computing prediction metrics")
                    train = user_ratings.iloc[partition_index[0]]
                    test = user_ratings.iloc[partition_index[1]]
                    test = remove_not_existent_items(test, self.__config.get_items_directory())

                    predictions = recsys.fit_eval_predict(user_id, train, test)
                    for metric in self.get_prediction_metric_list():
                        result_dict[str(metric)] = metric.perform(predictions, test)

                    prediction_metric_results.append(result_dict, ignore_index=True)

            prediction_metric_results = prediction_metric_results.groupby('from').mean().reset_index()

        # calculate ranking metrics
        if self.__ranking_metrics:
            if self.__config.get_ranking_algorithm() is None:
                raise ValueError("You must set ranking algorithm to compute ranking metrics")
            for user_id in user_id_list:
                logger.info("Computing ranking metrics for user %s" % user_id)
                user_ratings = self.__config.get_rating_frame()[
                    self.__config.get_rating_frame()['from_id'] == user_id]

                try:
                    self.__partitioning.set_dataframe(user_ratings)
                except ValueError:
                    continue

                self.__partitioning.set_dataframe(user_ratings)

                for partition_index in self.__partitioning:
                    result_dict = {}
                    train = user_ratings.iloc[partition_index[0]]
                    test = user_ratings.iloc[partition_index[1]]

                    truth = test.loc[:, 'to_id':'score']
                    truth.columns = ["to_id", "rating"]

                    recs_number = len(truth['rating'].values)
                    predictions = recsys.fit_eval_ranking(user_id, train, truth['to_id'].tolist(), recs_number)

                    for metric in self.get_ranking_metric_list():
                        result_dict['from'] = user_id
                        result_dict[str(metric)] = metric.perform(predictions, truth)

                    ranking_metric_results = ranking_metric_results.append(result_dict, ignore_index=True)
            ranking_metric_results = ranking_metric_results.groupby('from').mean().reset_index()

        serendipity_novelty_results = pd.DataFrame()
        fairness_metrics_results = []
        if self.__fairness_metrics or self.__serendipity_novelty_metrics:
            if isinstance(self.__config.get_score_prediction_algorithm(), ScorePredictionAlgorithm):
                raise ValueError("You must set ranking algorithm to compute fairness metrics")

            columns = ["from_id", "to_id", "rating"]
            score_frame = pd.DataFrame(columns=columns)
            for user_id in user_id_list:
                logger.info("User %s" % user_id)
                fit_result = recsys.fit_ranking(user_id, 10)

                fit_result_with_user = pd.DataFrame(columns=columns)
                fit_result.columns = ["to_id", "rating"]
                for i, row in fit_result.iterrows():
                    fit_result_with_user = pd.concat([fit_result_with_user, pd.DataFrame.from_records(
                        [(user_id, row["to_id"], row["rating"])], columns=columns)], ignore_index=True)

                score_frame = pd.concat([fit_result_with_user, score_frame], ignore_index=True)

            if self.__fairness_metrics:
                logger.info("Computing fairness metrics")
                for metric in self.get_fairness_metric_list():
                    fairness_metrics_results.append(metric.perform(score_frame, self.__config.get_rating_frame()))

            if self.__serendipity_novelty_metrics:
                logger.info("Computing novelty and serendipity")

        return prediction_metric_results, ranking_metric_results, fairness_metrics_results, serendipity_novelty_results
