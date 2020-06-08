import re

import pandas as pd
from typing import List

from orange_cb_recsys.recsys.algorithm import ScorePredictionAlgorithm, RankingAlgorithm
from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.utils.load_content import load_content_instance, get_unrated_items


class RecSys:
    def __init__(self, config: RecSysConfig):
        self.__config: RecSysConfig = config

    def __get_item_list(self, item_to_predict_id_list, user_ratings):
        if item_to_predict_id_list is None:
            # all items without rating if the list is not setted
            item_to_predict_list = get_unrated_items(self.__config.get_items_directory(), user_ratings)
        else:
            item_to_predict_list = [
                load_content_instance(self.__config.get_items_directory(), re.sub(r'[^\w\s]', '', item_id))
                for item_id in item_to_predict_id_list]

        return item_to_predict_list

    def fit_specific_items(self, user_id: str, item_to_predict_id_list: List[str] = None):
        if isinstance(self.__config.get_algorithm(), RankingAlgorithm):
            raise ValueError("You can't use ranking algorithms for predict score of specific items")

        # load user instance
        user = load_content_instance(self.__config.get_users_directory(), user_id)

        # load user ratings
        user_ratings = self.__config.get_rating_frame()[self.__config.get_rating_frame()['from_id'].str.match(user_id)]

        # define for which items calculate the prediction
        items = self.__get_item_list(item_to_predict_id_list, user_ratings)
        # calculate predictions
        score_frame = self.__config.get_algorithm().predict(items, user_ratings, self.__config.get_items_directory())

        return score_frame

    def fit(self, user_id: str, recs_number: int):
        if isinstance(self.__config.get_algorithm(), ScorePredictionAlgorithm):
            raise ValueError("You can't use rating prediction algorithms for this method")

        # load user instance
        user = load_content_instance(self.__config.get_users_directory(), user_id)

        # load user ratings
        user_ratings = self.__config.get_rating_frame()[self.__config.get_rating_frame()['from_id'].str.match(user_id)]

        # calculate predictions
        score_frame = self.__config.get_algorithm().predict(user_ratings, recs_number,
                                                            self.__config.get_items_directory())

        return score_frame

    def fit_eval(self, user_id: str, user_ratings: pd.DataFrame, test_set: pd.DataFrame):
        # load user instance
        user = load_content_instance(self.__config.get_users_directory(), user_id)

        score_frame = None
        if isinstance(self.__config.get_algorithm(), ScorePredictionAlgorithm):
            # get test set items
            item_to_predict_id_list = [item for item in test_set.item_id]  # unrated items list
            items = [load_content_instance(self.__config.get_items_directory(), re.sub(r'[^\w\s]', '', item_id))
                     for item_id in item_to_predict_id_list]

            # calculate predictions
            score_frame = self.__config.get_algorithm().predict(items, user_ratings,
                                                                self.__config.get_items_directory())
        elif isinstance(self.__config.get_algorithm(), RankingAlgorithm):
            score_frame = self.__config.get_algorithm().predict(user_ratings, test_set.shape[0],
                                                                self.__config.get_items_directory())

        return score_frame
