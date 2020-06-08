import os
import pandas as pd

from orange_cb_recsys.evaluation.metrics import perform_prediction_metrics, perform_ranking_metrics, \
    perform_fairness_metrics
from orange_cb_recsys.evaluation.partitioning import Partitioning
from orange_cb_recsys.recsys.algorithm import RankingAlgorithm, ScorePredictionAlgorithm
from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.recsys.recsys import RecSys


class EvalModel:
    def __init__(self, config: RecSysConfig,
                 partitioning: Partitioning,
                 prediction_metric: bool = True,
                 ranking_metric: bool = True,
                 fairness_metric: bool = True):
        self.__config: RecSysConfig = config
        self.__partitioning = partitioning
        self.__prediction_metric = prediction_metric
        self.__ranking_metric = ranking_metric
        self.__fairness_metric = fairness_metric

    def fit(self):
        # initialize recommender to call for prediction calcs
        recsys = RecSys(self.__config)

        # get all users in specified directory
        user_id_list = [os.path.splitext(filename)[0] for filename in os.listdir(self.__config.get_users_directory())]

        # define results structure
        prediction_metric_results = pd.DataFrame(columns=["from", "RMSE", "MAE", "serendipity", "novelty"])
        ranking_metric_results = \
            pd.DataFrame(columns=["from", "Precision", "Recall", "F1", "NDCG", "pearson", "kendall", "spearman"])

        # calculate prediction metrics
        if self.__prediction_metric:
            if isinstance(self.__config.get_algorithm(), RankingAlgorithm):
                raise ValueError("Can't calculate predictions metrics for ranking algorithm results")

            for user_id in user_id_list:
                user_ratings = self.__config.get_rating_frame()[
                    self.__config.get_rating_frame()['from_id'].str.match(user_id)]

                try:
                    self.__partitioning.set_dataframe(user_ratings)
                except ValueError:
                    continue

                for partition_index in self.__partitioning:
                    train = user_ratings.iloc[partition_index[0]]
                    test = user_ratings.iloc[partition_index[1]]

                    predictions = pd.Series(recsys.fit_eval(train, test).rating, name="rating")
                    truth = pd.Series(test.score.values, name="rating")

                    result_dict = perform_prediction_metrics(predictions, truth)
                    prediction_metric_results.append(result_dict)

                prediction_metric_results = prediction_metric_results.groupby('from').mean()

        # calculate ranking metrics
        if self.__ranking_metric:
            if isinstance(self.__config.get_algorithm(), ScorePredictionAlgorithm):
                raise ValueError("Can't calculate predictions metrics for ranking algorithm results")

            for user_id in user_id_list:
                user_ratings = self.__config.get_rating_frame()[
                    self.__config.get_rating_frame()['from_id'].str.match(user_id)]

                try:
                    self.__partitioning.set_dataframe(user_ratings)
                except ValueError:
                    continue

                self.__partitioning.set_dataframe(user_ratings)

                for partition_index in self.__partitioning:
                    train = user_ratings.iloc[partition_index[0]]
                    test = user_ratings.iloc[partition_index[1]]

                    predictions = recsys.fit_eval(train, test)
                    truth = pd.DataFrame(test[test.columns[[1, 2]]]).reset_index(drop=True)
                    truth.columns = ["to_id", "rating"]

                    print(predictions, truth)

                    result_dict = perform_ranking_metrics(predictions, truth)
                    result_dict["from"] = user_id
                    ranking_metric_results = ranking_metric_results.append(result_dict, ignore_index=True)

                ranking_metric_results.groupby('from').mean()

        # calculate fairness metrics
        fairness_metrics_results = \
            pd.DataFrame(columns=["from", "gini-index", "delta-gaps", "pop_ratio_profile_vs_recs", "pop_recs_correlation", "recs_long_tail_distr"])
        if self.__fairness_metric:
            columns = ["from_id", "to_id", "rating"]
            score_frame = pd.DataFrame(columns=columns)
            for user_id in user_id_list:
                user_ratings = self.__config.get_rating_frame()[
                    self.__config.get_rating_frame()['from_id'].str.match(user_id)]

                if isinstance(self.__config.get_algorithm(), ScorePredictionAlgorithm):
                    fit_result = recsys.fit_predict(user_id)
                else:
                    fit_result = recsys.fit_ranking(user_id, 20)
                fit_result_with_user = pd.DataFrame(columns=columns)

                for i, row in fit_result.iterrows():
                    fit_result_with_user = pd.concat([fit_result_with_user, pd.DataFrame.from_records(
                        [(user_id, row["to_id"], row["rating"])], columns=columns)], ignore_index=True)

                score_frame = pd.concat([fit_result_with_user, score_frame], ignore_index=True)

                try:
                    self.__partitioning.set_dataframe(user_ratings)
                except ValueError:
                    continue

            fairness_metrics_results = perform_fairness_metrics(score_frame=score_frame,
                                                                truth_frame=self.__config.get_rating_frame(),
                                                                algorithm_name=str(self.__config.get_algorithm()))

        return prediction_metric_results, ranking_metric_results, fairness_metrics_results
