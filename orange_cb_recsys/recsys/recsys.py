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

    def __get_item_to_predict_id_list(self, item_to_predict_id_list, user_ratings):
        if item_to_predict_id_list is None:
            item_list = [os.path.splitext(filename)[0] for filename in os.listdir(self.__config.get_items_directory())]
            try:
                # list of items without rating
                item_to_predict_id_list = [item for item in item_list if not user_ratings['to_id'].str.contains(item).any()]
            except KeyError:
                item_to_predict_id_list = item_list

        return item_to_predict_id_list

    def __predict_item_list(self, user, user_ratings, item_to_predict_id_list):
        columns = ["item_id", "rating"]
        score_frame = pd.DataFrame(columns=columns)

        for item_id in item_to_predict_id_list:
            # load item instance
            item = self.__load_content_instance(self.__config.get_items_directory(), item_id)

            predicted_rating = self.__predict_item(user, item, user_ratings)

            score_frame = pd.concat([pd.DataFrame.from_records([(item_id, predicted_rating)], columns=columns), score_frame], ignore_index=True)

        return score_frame

    def __predict_item(self, user, item, user_ratings):
        if isinstance(self.__config.get_score_prediction_algorithm(), RatingsSPA):
            assert (self.__config.get_rating_frame() is not None), \
                "You must specify where to find ratings if you use ratings based algorithm"

            predicted_rating = self.__config.get_score_prediction_algorithm(). \
                predict(item, user_ratings, self.__config.get_items_directory())

        else:
            predicted_rating = self.__config.get_score_prediction_algorithm().predict(user, item)

        return predicted_rating

    def __load_content_instance(self, directory, content_id):
        content_filename = directory + '/' + content_id + '.bin'
        with open(content_filename, "rb") as content_file:
            content: Content = pickle.load(content_file)

        return content

    def fit(self, user_id: str, item_to_predict_id_list: List[str] = None, rank: bool = False):
        # load user instance
        user = self.__load_content_instance(self.__config.get_users_directory(), user_id)

        # load user ratings
        user_ratings = self.__config.get_rating_frame()[self.__config.get_rating_frame()['from_id'].str.match(user_id)]

        # define for which items calculate the prediction
        item_to_predict_id_list = self.__get_item_to_predict_id_list(item_to_predict_id_list, user_ratings)

        # calculate predictions
        score_frame = self.__predict_item_list(user, user_ratings, item_to_predict_id_list)

        if rank:
            return self.__config.get_ranking_algorithm().rank(score_frame)
        else:
            return score_frame

    def fit_eval(self, user_id: str, user_ratings: pd.DataFrame, test_set: pd.DataFrame, rank: bool = False):
        # load user instance
        user = self.__load_content_instance(self.__config.get_users_directory(), user_id)

        # get test set items
        item_to_predict_id_list = [item for item in test_set.item_id]  # lista di item non valutati

        # calculate predictions
        score_frame = self.__predict_item_list(user, user_ratings, item_to_predict_id_list)

        if rank:
            return self.__config.get_ranking_algorithm().rank(score_frame)
        else:
            return score_frame
