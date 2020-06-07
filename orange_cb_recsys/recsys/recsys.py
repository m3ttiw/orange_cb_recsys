import os
import re

import pandas as pd
from typing import List

from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.recsys.score_prediction_algorithms.index_query import IndexQuery
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import RatingsSPA
from orange_cb_recsys.utils.load_content import load_content_instance


class RecSys:
    def __init__(self, config: RecSysConfig):
        self.__config: RecSysConfig = config

    def __get_item_list(self, item_to_predict_id_list, user_ratings):
        if item_to_predict_id_list is None:
            directory_file_list = [os.path.splitext(filename)[0]
                                   for filename in os.listdir(self.__config.get_items_directory())
                                   if filename != 'search_index']

            directory_item_list = [load_content_instance(self.__config.get_items_directory(), item_filename) for
                                   item_filename in directory_file_list]

            # list of items without rating
            item_to_predict_list = [item for item in directory_item_list if
                                    not user_ratings['to_id'].str.contains(item.get_content_id()).any()]
        else:
            item_to_predict_list = [
                load_content_instance(self.__config.get_items_directory(), re.sub(r'[^\w\s]', '', item_id))
                for item_id in item_to_predict_id_list]

        return item_to_predict_list

    def __predict_item_list(self, user, user_ratings, items):
        if isinstance(self.__config.get_score_prediction_algorithm(), RatingsSPA):
            score_frame = self.__config.get_score_prediction_algorithm(). \
                predict(items, user_ratings, self.__config.get_items_directory())

        else:
            score_frame = self.__config.get_score_prediction_algorithm().predict(user, items)

        return score_frame

    def __fit(self, user_id, item_to_predict_id_list, rank):
        if isinstance(self.__config.get_score_prediction_algorithm(), RatingsSPA) or \
                isinstance(self.__config.get_score_prediction_algorithm(), IndexQuery):
            if self.__config.get_rating_frame() is None:
                raise ValueError("You must set ratings frame if you want to use "
                                 "ratings based algorithm")

        # load user instance
        user = load_content_instance(self.__config.get_users_directory(), user_id)

        # load user ratings
        user_ratings = self.__config.get_rating_frame()[self.__config.get_rating_frame()['from_id'].str.match(user_id)]

        if isinstance(self.__config.get_score_prediction_algorithm(), IndexQuery):
            score_frame = self.__config.get_score_prediction_algorithm(). \
                predict(user_ratings, self.__config.get_items_directory())
        else:
            # define for which items calculate the prediction
            items = self.__get_item_list(item_to_predict_id_list, user_ratings)
            # calculate predictions
            score_frame = self.__predict_item_list(user, user_ratings, items)

        if rank:
            return self.__config.get_ranking_algorithm().rank(score_frame)
        else:
            return score_frame

    def fit_specific_items(self, user_id: str, item_to_predict_id_list: List[str] = None, rank: bool = False):
        if isinstance(self.__config.get_score_prediction_algorithm(), IndexQuery):
            raise ValueError("You can't use this recommender for predict score of specific items")

        return self.__fit(user_id, item_to_predict_id_list, rank)

    def fit(self, user_id: str, rank: bool = False):
        return self.__fit(user_id, None, rank)

    def fit_eval(self, user_id: str, user_ratings: pd.DataFrame, test_set: pd.DataFrame, rank: bool = False):
        # load user instance
        user = load_content_instance(self.__config.get_users_directory(), user_id)

        if isinstance(self.__config.get_score_prediction_algorithm(), IndexQuery):
            score_frame = self.__config.get_score_prediction_algorithm(). \
                predict(user_ratings, self.__config.get_items_directory())
        else:
            # get test set items
            item_to_predict_id_list = [item for item in test_set.item_id]  # lista di item non valutati
            items = [load_content_instance(self.__config.get_items_directory(), re.sub(r'[^\w\s]', '', item_id))
                     for item_id in item_to_predict_id_list]

            # calculate predictions
            score_frame = self.__predict_item_list(user, user_ratings, items)

        if rank:
            return self.__config.get_ranking_algorithm().rank(score_frame)
        else:
            return score_frame
