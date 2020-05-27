import os
import pickle
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
        if item_to_predict_filename_list is None:
            item_list = self.__get_item_filename_list()
            try:
                ratings_values = user.get_field(self.__config.get_rating_field()).get_representation(str(0))
                item_to_predict_filename_list = [item for item in item_list if item not in ratings_values.get_value().keys()]  # lista di item non valutati
            except KeyError:
                item_to_predict_filename_list = item_list

        score_dict = {}
        for item_filename in item_to_predict_filename_list:
            item = pickle.load(item_filename)

            if isinstance(self.__config.get_score_prediction_algorithm(), RatingsSPA):
                assert (self.__config.get_rating_field() is not None), "You must specify where to find ratings if you use ratings based algorithm"
                score_dict[item_filename] = self.__config.get_score_prediction_algorithm().\
                    predict(user, item, self.__config.get_rating_field(), self.__config.get_items_directory())
            else:
                score_dict[item_filename] = self.__config.get_score_prediction_algorithm().predict(user, item)
        if rank:
            return self.__config.get_ranking_algorithm().rank(score_dict)
        else:
            return score_dict
