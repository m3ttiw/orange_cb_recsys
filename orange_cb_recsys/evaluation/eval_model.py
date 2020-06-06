import os
import pandas as pd

from orange_cb_recsys.evaluation.metrics import perform_prediction_metrics, perform_ranking_metrics, \
    perform_fairness_metrics
from orange_cb_recsys.evaluation.partitioning import Partitioning
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
        prediction_metric_results = pd.DataFrame(columns=["user", "RMSE", "MAE", "serendipity", "novelty"])
        ranking_metric_results = pd.DataFrame(columns=["user", "precision", "recall", "F1", "MAE"])

        # calculate prediction metrics
        if self.__prediction_metric:
            for user_id in user_id_list:
                user_ratings = self.__config.get_rating_frame()[
                    self.__config.get_rating_frame()['user_id'].str.match(user_id)]

                try:
                    self.__partitioning.set_dataframe(user_ratings)
                except ValueError:
                    continue

                for partition_index in self.__partitioning:
                    train = user_ratings.iloc[partition_index[0]]
                    test = user_ratings.iloc[partition_index[1]]

                    predictions = pd.Series(recsys.fit_eval(user_id, train, test).rating, name="rating")
                    truth = pd.Series(test.derived_score.values, name="rating")

                    result_dict = perform_prediction_metrics(predictions, truth)
                    prediction_metric_results = pd.concat([
                        pd.DataFrame.from_records([(user_id, result_dict["RMSE"], result_dict["MAE"])],
                                                  columns=["user", "RMSE", "MAE"]),
                        prediction_metric_results], ignore_index=True)

                prediction_metric_results = prediction_metric_results.groupby('user').mean()

        # calculate ranking metrics
        if self.__ranking_metric:
            for user_id in user_id_list:
                user_ratings = self.__config.get_rating_frame()[
                    self.__config.get_rating_frame()['user_id'].str.match(user_id)]

                try:
                    self.__partitioning.set_dataframe(user_ratings)
                except ValueError:
                    continue

                self.__partitioning.set_dataframe(user_ratings)

                for partition_index in self.__partitioning:
                    train = user_ratings.iloc[partition_index[0]]
                    test = user_ratings.iloc[partition_index[1]]

                    predictions = recsys.fit_eval(user_id, train, test, True)
                    truth = self.__config.get_ranking_algorithm().rank(test)

                    result_dict = perform_ranking_metrics(predictions, truth)
                    ranking_metric_results = pd.concat([
                        pd.DataFrame.from_records([(user_id, result_dict["precision"], result_dict["recall"],
                                                    result_dict["F1"], result_dict["NDCG"])],
                                                  columns=["user", "precision", "recall", "F1", "MAE"]),
                        ranking_metric_results], ignore_index=True)

                ranking_metric_results.groupby('user').mean()

        # calculate fairness metrics
        fairness_metrics_results = pd.DataFrame(columns=["user", "gini-index", "delta-gaps", "pop_ratio_profile_vs_recs", "pop_recs_correlation", "recs_long_tail_distr"])
        if self.__fairness_metric:
            columns = ["user_id", "item_id", "rating"]
            score_frame = pd.DataFrame(columns=columns)
            for user_id in user_id_list:
                user_ratings = self.__config.get_rating_frame()[
                    self.__config.get_rating_frame()['user_id'].str.match(user_id)]

                fit_result = recsys.fit(user_id)
                fit_result_with_user = pd.DataFrame(columns=columns)

                for i, row in fit_result.iterrows():
                    fit_result_with_user = pd.concat([fit_result_with_user, pd.DataFrame.from_records(
                        [(user_id, row["item_id"], row["rating"])], columns=columns)], ignore_index=True)

                score_frame = pd.concat([fit_result_with_user, score_frame], ignore_index=True)

                try:
                    self.__partitioning.set_dataframe(user_ratings)
                except ValueError:
                    continue

            print(score_frame)
            fairness_metrics_results = perform_fairness_metrics(score_frame)

        return prediction_metric_results, ranking_metric_results, fairness_metrics_results
