import os
import pickle
from typing import List
import pandas as pd

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.evaluation.metrics import perform_prediction_metrics, perform_ranking_metrics
from orange_cb_recsys.evaluation.partitioning import Partitioning
from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.recsys.ranking_algorithms.ranking_algorithm import TopNRanking
from orange_cb_recsys.recsys.recsys import RecSys
from orange_cb_recsys.recsys.score_prediction_algorithms.ratings_based import CentroidVector


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

    def __get_user_filename_list(self) -> List[dir]:
        return os.listdir(self.__config.get_users_directory())

    def __create_ratings_dataframe(self, user: str) -> pd.DataFrame:
        columns = ["user_id", "item_id", "rating"]
        ratings = pd.DataFrame(columns=columns)

        user: Content = pickle.load(self.__config.get_users_directory() + user)
        user_ratings = user.get_field(self.__config.get_rating_field()).get_representation(str(0)).get_value()
        for item_id, score in user_ratings.items():
            ratings = pd.concat([pd.DataFrame([[user, item_id, score]], columns=columns), ratings], ignore_index=True)

        return ratings

    def fit(self):
        recsys = RecSys(self.__config)
        user_filename_list = self.__get_user_filename_list()

        prediction_metric_results = pd.DataFrame(columns=["user", "rmse", "mae"])
        ranking_metric_results = pd.DataFrame(columns=["user", "precision", "recall", "F1", "MAE"])

        if self.__prediction_metric:
            for user in user_filename_list:
                ratings = self.__create_ratings_dataframe(user)
                self.__partitioning.set_dataframe(ratings)

                for partition_index in self.__partitioning:
                    train = ratings.iloc[partition_index[0]]
                    test = ratings.iloc[partition_index[1]]

                    predictions = pd.Series(recsys.fit_eval(user, train).rating)
                    truth = pd.Series(test.rating)

                    result_dict = perform_prediction_metrics(predictions, truth)
                    prediction_metric_results = pd.concat(pd.DataFrame.from_records([(user, result_dict["rmse"], result_dict["mae"])]),
                                                          prediction_metric_results, ignore_index=True)

                prediction_metric_results.groupby('user').mean()

        if self.__ranking_metric:
            for user in user_filename_list:
                ratings = self.__create_ratings_dataframe(user)
                self.__partitioning.set_dataframe(ratings)

                for partition_index in self.__partitioning:
                    train = ratings.iloc[partition_index[0]]
                    test = ratings.iloc[partition_index[1]]

                    predictions = recsys.fit_eval(user, train, True)
                    truth = self.__config.get_ranking_algorithm().rank(test)

                    result_dict = perform_ranking_metrics(predictions, truth)
                    ranking_metric_results = pd.concat(
                        pd.DataFrame.from_records([(user, result_dict["precision"], result_dict["recall"], result_dict["F1"], result_dict["NDCG"])]),
                        ranking_metric_results, ignore_index=True)

                ranking_metric_results.groupby('user').mean()

        return prediction_metric_results, ranking_metric_results



SPA = CentroidVector("Plot")
recsys_config = RecSysConfig("users", "items", SPA, TopNRanking(15), "ratings")
model = EvalModel(recsys_config, True, True, False)
frame, frame2 = model.fit()
