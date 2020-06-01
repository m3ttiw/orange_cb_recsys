import os
import pickle
import pandas as pd
from typing import List

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import RatingsSPA


class RecSys:
    def __init__(self, config: RecSysConfig):
        self.__config: RecSysConfig = config

    def __get_item_filename_list(self) -> List[dir]:
        return os.listdir(self.__config.get_items_directory())

    def fit(self, user_filename: str, item_to_predict_filename_list: List[str] = None, rank: bool = False):
        user: Content = pickle.load(self.__config.get_users_directory() + user_filename)

        user_ratings = self.__config.get_rating_frame()[self.__config.get_rating_frame()['user_id'].str.match(user_filename)]

        if item_to_predict_filename_list is None:
            item_list = self.__get_item_filename_list()
            try:
                # list of items without rating
                item_to_predict_filename_list = [item for item in item_list if
                                                 not user_ratings['item_id'].str.contains(item).any()]
            except KeyError:
                item_to_predict_filename_list = item_list

        score_frame = pd.Dataframe(columns=["item", "rating"])
        for item_filename in item_to_predict_filename_list:
            item = pickle.load(item_filename)

            if isinstance(self.__config.get_score_prediction_algorithm(), RatingsSPA):
                assert (self.__config.get_rating_frame() is not None), \
                    "You must specify where to find ratings if you use ratings based algorithm"

                predicted_rating = self.__config.get_score_prediction_algorithm().\
                    predict(user, item, user_ratings, self.__config.get_items_directory())

            else:
                predicted_rating = self.__config.get_score_prediction_algorithm().predict(user, item)

            score_frame = pd.concat(pd.DataFrame.from_records([(item_filename, predicted_rating)]), score_frame,
                                    ignore_index=True)

        if rank:
            return self.__config.get_ranking_algorithm().rank(score_frame)
        else:
            return score_frame

    def fit_eval(self, user_filename: str, user_ratings: pd.DataFrame, rank: bool = False):
        user: Content = pickle.load(self.__config.get_users_directory() + user_filename)
        item_list = self.__get_item_filename_list()
        item_to_predict_filename_list = [item for item in item_list if
                                         item not in user_ratings.item_id]  # lista di item non valutati

        score_frame = pd.Dataframe(columns=["item", "rating"])
        for item_filename in item_to_predict_filename_list:
            item = pickle.load(item_filename)

            if isinstance(self.__config.get_score_prediction_algorithm(), RatingsSPA):
                predicted_rating = self.__config.get_score_prediction_algorithm(). \
                    predict(user, item, user_ratings, self.__config.get_items_directory())
            else:
                predicted_rating = self.__config.get_score_prediction_algorithm().predict(user, item)

            score_frame = pd.concat(pd.DataFrame.from_records([(item_filename, predicted_rating)]), score_frame,
                                    ignore_index=True)

        if rank:
            return self.__config.get_ranking_algorithm().rank(score_frame)
        else:
            return score_frame
