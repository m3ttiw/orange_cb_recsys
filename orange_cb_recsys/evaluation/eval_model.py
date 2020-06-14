import os
from typing import List, Dict

import pandas as pd

from orange_cb_recsys.evaluation.metrics import perform_prediction_metrics, perform_ranking_metrics, \
    perform_fairness_metrics, logger, perform_serendipity_novelty_metrics
from orange_cb_recsys.evaluation.partitioning import Partitioning
from orange_cb_recsys.recsys.algorithm import RankingAlgorithm, ScorePredictionAlgorithm
from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.recsys.recsys import RecSys
from orange_cb_recsys.utils.load_content import remove_not_existent_items


class FairnessMetricsConfig:
    """
    Configuration for the fairness computation
    Args:
        output_directory (str):
        user_groups: Groups of users, key = group name, value = percentage of users
    """
    def __init__(self, output_directory: str, user_groups: Dict[str, float]):
        # algorithm name automatically retrieved in eval model
        self.__output_directory: str = output_directory
        self.__user_groups: Dict[str, float] = user_groups

    def get_directory(self):
        return self.__output_directory

    def get_user_groups(self):
        return self.__user_groups


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
        partitioning (Partitioning): Partitions
        prediction_metric (bool): Whether you want to evaluate the rating prediction phase
        ranking_metrics_config (RankingMetricsConfig): Configuration for ranking metrics copmuting
        fairness_metric_config (FairnessMetricsConfig): Configuration for the fairness computation
    """
    def __init__(self, config: RecSysConfig,
                 partitioning: Partitioning,
                 prediction_metric: bool = True,
                 ranking_metrics_config: RankingMetricsConfig = None,
                 fairness_metric_config: FairnessMetricsConfig = None,
                 serendipity_novelty_metrics: bool = False):
        self.__config: RecSysConfig = config
        self.__partitioning = partitioning
        self.__prediction_metric = prediction_metric
        self.__serendipity_novelty_metrics = serendipity_novelty_metrics
        self.__ranking_metrics_config: RankingMetricsConfig = ranking_metrics_config
        self.__fairness_metric_config: FairnessMetricsConfig = fairness_metric_config

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
        # initialize recommender to call for prediction calcs
        recsys = RecSys(self.__config)

        # get all users in specified directory
        logger.info("Loading user instances")
        user_id_list = [os.path.splitext(filename)[0] for filename in os.listdir(self.__config.get_users_directory())]

        # define results structure
        prediction_metric_results = pd.DataFrame(columns=["from", "RMSE", "MAE", "serendipity", "novelty"])
        ranking_metric_results = \
            pd.DataFrame(columns=["from", "Precision", "Recall", "F1", "NDCG", "pearson", "kendall", "spearman"])

        # calculate prediction metrics
        if self.__prediction_metric:
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
                    train = user_ratings.iloc[partition_index[0]]
                    test = user_ratings.iloc[partition_index[1]]
                    test = remove_not_existent_items(test, self.__config.get_items_directory())

                    predictions = pd.Series(recsys.fit_eval_predict(user_id, train, test).rating,
                                            name="rating", dtype=float)
                    truth = pd.Series(test.score.values, name="rating", dtype=float)

                    logger.info("Computing metrics")
                    result_dict = perform_prediction_metrics(predictions, truth)
                    result_dict['from'] = user_id
                    prediction_metric_results = prediction_metric_results.append(result_dict,
                                                                                 ignore_index=True)

            prediction_metric_results = prediction_metric_results.groupby("from").mean().reset_index()

        # calculate ranking metrics
        if self.__ranking_metrics_config is not None:
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
                    train = user_ratings.iloc[partition_index[0]]
                    test = user_ratings.iloc[partition_index[1]]

                    truth = test.loc[:, 'to_id':'score']
                    truth.columns = ["to_id", "rating"]

                    relevant_items_number = len(truth[truth['rating'] >= self.__ranking_metrics_config.get_relevant_threshold()].to_id.tolist())

                    predictions = recsys.fit_eval_ranking(user_id, train, truth['to_id'].tolist(), relevant_items_number)
                    result_dict = perform_ranking_metrics(predictions, truth,
                                                          relevant_threshold=self.__ranking_metrics_config.get_relevant_threshold(),
                                                          relevance_split=self.__ranking_metrics_config.get_relevance_split())

                    result_dict["from"] = user_id
                    ranking_metric_results = ranking_metric_results.append(result_dict, ignore_index=True)

                ranking_metric_results = ranking_metric_results.groupby('from').mean().reset_index()

        # calculate fairness metrics
        fairness_metrics_results = \
            pd.DataFrame(columns=["from", "gini-index", "delta-gaps", "pop_ratio_profile_vs_recs",
                                  "pop_recs_correlation", "recs_long_tail_distr"])

        serendipity_novelty_results = \
            pd.DataFrame(columns=["from", "serendipity", "novelty"])

        if self.__fairness_metric_config is not None or self.__serendipity_novelty_metrics:
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

            if self.__fairness_metric_config is not None:
                logger.info("Computing fairness metrics")
                fairness_metrics_results = perform_fairness_metrics(score_frame=score_frame,
                                                                    user_groups=self.__fairness_metric_config.get_user_groups(),
                                                                    truth_frame=self.__config.get_rating_frame(),
                                                                    algorithm_name='test',
                                                                    file_output_directory=self.__fairness_metric_config.get_directory())

            if self.__serendipity_novelty_metrics:
                logger.info("Computing novelty and serendipity")
                serendipity_novelty_results = perform_serendipity_novelty_metrics(score_frame, self.__config.get_rating_frame())

        return prediction_metric_results, ranking_metric_results, fairness_metrics_results, serendipity_novelty_results
