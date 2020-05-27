import os
import pickle
from typing import List
import pandas as pd

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.evaluation.metrics import perform_prediction_metrics
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

    def __get_user_filename_list(self) -> List[dir]:
        return os.listdir(self.__config.get_users_directory())

    def __create_ratings_dataframe(self) -> pd.DataFrame:
        user_filename_list = self.__get_user_filename_list()
        columns = ["user_id", "item_id", "rating"]
        ratings = pd.DataFrame(columns=columns)
        for user_filename in user_filename_list:
            user: Content = pickle.load(self.__config.get_users_directory() + user_filename)
            user_ratings = user.get_field(self.__config.get_rating_field()).get_representation(str(0)).get_value()
            for item_id, score in user_ratings.items():
                ratings = pd.concat([pd.DataFrame([[user_filename, item_id, score]], columns=columns), ratings],
                                    ignore_index=True)

        return ratings

    def fit(self):
        recsys = RecSys(self.__config)
        ratings = self.__create_ratings_dataframe()
        self.__partitioning.set_dataframe(ratings)

        if self.__prediction_metric:
            predictions = pd.Series()
            truth = pd.Series()
            for partition_index in self.__partitioning:
                train = ratings.iloc[partition_index[0]]
                test = ratings.iloc[partition_index[1]]

                for user, item, truth_rating in zip(test.user, test.item, test.rating):
                    predict_score = recsys.fit(user, item).values()[0]
                    predictions.append(pd.Series([predict_score]), ignore_index=True)
                    truth.append(pd.Series([truth_rating]), ignore_index=True)

            perform_prediction_metrics(predictions, truth)

        if self.__ranking_metric:
            predictions = pd.DataFrame(columns=["item_id", "rating"])
            truth = pd.DataFrame(columns=["item_id", "rating"])

            for partition_index in self.__partitioning:
                train = ratings.iloc[partition_index[0]]
                test = ratings.iloc[partition_index[1]]

                for user in test.user:
                    item_to_predict_list = [item for item in test[test['user_id'].str.match(user)].item]
                    predicted_ranking = recsys.fit(user, item_to_predict_list)
                    predictions.concat(pd.DataFrame([predict_score]), ignore_index=True)
                    truth.append(pd.Series([truth_rating]), ignore_index=True)

            perform_prediction_metrics(predictions, truth)
