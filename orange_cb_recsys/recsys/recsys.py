import re

import pandas as pd
from typing import List

from orange_cb_recsys.recsys.algorithm import ScorePredictionAlgorithm, RankingAlgorithm
from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.load_content import load_content_instance, get_unrated_items


class RecSys:
    """
    Class that represent a recommender system
    Args:
        config (RecSysConfig): Configuration of the recommender system
    """

    def __init__(self, config: RecSysConfig):
        self.__config: RecSysConfig = config

    def __get_item_list(self, item_to_predict_id_list, user_ratings):
        if item_to_predict_id_list is None:
            # all items without rating if the list is not set
            item_to_predict_list = get_unrated_items(self.__config.get_items_directory(), user_ratings)
        else:
            item_to_predict_list = [
                load_content_instance(self.__config.get_items_directory(), re.sub(r'[^\w\s]', '', item_id))
                for item_id in item_to_predict_id_list]

        return item_to_predict_list

    def fit_predict(self, user_id: str, item_to_predict_id_list: List[str] = None):
        """
        Computes the predicted rating for specified user and items,
        should be used when a score prediction algorithm (instead of a ranking algorithm)
        was chosen in the config

        Args:
            user_id: user for which calculate the predictions
            item_to_predict_id_list: items for which the prediction will be computed,
                if None all unrated items will be used
        Returns:
            score_frame (DataFrame): result frame whose columns are: to_id, rating

        Raises:
             ValueError: if the algorithm is a ranking algorithm
        """
        if self.__config.get_score_prediction_algorithm() is None:
            raise ValueError("You must set score prediction algorithm to use this method")

        # load user ratings
        logger.info("Loading user ratings")
        user_ratings = self.__config.get_rating_frame()[self.__config.get_rating_frame()['from_id'] == user_id]
        user_ratings = user_ratings.sort_values(['to_id'], ascending=True)

        # define for which items calculate the prediction
        logger.info("Defining for which items the prediction will be computed")
        items = self.__get_item_list(item_to_predict_id_list, user_ratings)

        # calculate predictions
        logger.info("Computing predicitons")
        score_frame = self.__config.get_score_prediction_algorithm().predict(user_id, items, user_ratings,
                                                                             self.__config.get_items_directory())

        return score_frame

    def fit_ranking(self, user_id: str, recs_number: int):
        """
        Computes the predicted rating for specified user and items,
        should be used when a  ranking algorithm (instead of a score prediction algorithm)
        was chosen in the config

        Args:
            user_id: user for which calculate the predictions
            recs_number: how many items should the returned ranking contain,
            the ranking length can be lower
        Returns:
            score_frame (DataFrame): result frame whose columns are: to_id, rating

        Raises:
             ValueError: if the algorithm is a score prediction algorithm
        """
        if self.__config.get_ranking_algorithm() is None:
            raise ValueError("You must set ranking algorithm to use this method")

        # load user ratings
        logger.info("Loading user ratings")
        user_ratings = self.__config.get_rating_frame()[self.__config.get_rating_frame()['from_id'] == user_id]
        user_ratings = user_ratings.sort_values(['to_id'], ascending=True)

        # calculate predictions
        logger.info("Computing ranking")
        score_frame = self.__config.get_ranking_algorithm().predict(user_id, user_ratings, recs_number,
                                                                    self.__config.get_items_directory())

        return score_frame

    def fit_eval_predict(self, user_id, user_ratings: pd.DataFrame, test_set: pd.DataFrame):
        """
        Computes predicted ratings, or ranking (according to algorithm chosen in the config)
        user ratings will be used as train set to fit the algorithm.
        If the algorithm is score_prediction the rating for the item in the test set will
        be predicted, else a ranking with recs_number = len(test_set) will be computed

        Args:
            user_ratings: train set
            test_set:
        Returns:
            score_frame (DataFrame): result frame whose columns are: to_id, rating
        """
        logger.info("Loading items")
        item_to_predict_id_list = [item for item in test_set.to_id]  # unrated items list
        items = [load_content_instance(self.__config.get_items_directory(), re.sub(r'[^\w\s]', '', item_id))
                 for item_id in item_to_predict_id_list]

        logger.info("Loaded %d items" % len(items))

        # calculate predictions
        logger.info("Computing predictions")
        score_frame = self.__config.get_score_prediction_algorithm().predict(user_id, items, user_ratings,
                                                                             self.__config.get_items_directory())

        return score_frame

    def fit_eval_ranking(self, user_id, user_ratings: pd.DataFrame, test_set: pd.DataFrame):
        """
        Computes a ranking, using as training set the ratings provided by he user

        Args:
            user_id:
            user_ratings (pd.DataFrame): Training set
            test_set (pd.DataFrame)
        """
        score_frame = self.__config.get_ranking_algorithm().predict(user_id, user_ratings, test_set.shape[0],
                                                                    self.__config.get_items_directory())

        return score_frame
