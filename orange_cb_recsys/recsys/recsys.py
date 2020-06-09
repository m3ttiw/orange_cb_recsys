import re

import pandas as pd
from typing import List

from orange_cb_recsys.recsys.algorithm import ScorePredictionAlgorithm, RankingAlgorithm
from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.load_content import load_content_instance, get_unrated_items


class RecSys:
    """
    Class that represent a recommender system, to isntatiate
    this class a config object must be provided
    """
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

    def fit_predict(self, user_id: str, item_to_predict_id_list: List[str] = None):
        """
        Computes the predicted rating for specified user and items,
        should be used when a score prediction algorithm (instead of a ranking algorithm)
        was chosen in the config

        Args
            user_id: user for which calculate the predictions
            item_to_predict_id_list: item for which the prediction will be computed,
                if None all unrated items will be used
        Returns:
            (DataFrame): result frame whose columns are: to_id, rating

        Raises:
             ValueError: if the algorithm is a ranking algorithm
        """
        if isinstance(self.__config.get_algorithm(), RankingAlgorithm):
            raise ValueError("You can't use ranking algorithms for predict score of specific items")

        # load user ratings
        logger.info("Loading user ratings")
        user_ratings = self.__config.get_rating_frame()[self.__config.get_rating_frame()['from_id'].str.match(user_id)]

        # define for which items calculate the prediction
        logger.info("Defining for which items the prediction will be computed")
        items = self.__get_item_list(item_to_predict_id_list, user_ratings)

        # calculate predictions
        logger.info("Computing predicitons")
        score_frame = self.__config.get_algorithm().predict(items, user_ratings, self.__config.get_items_directory())

        return score_frame

    def fit_ranking(self, user_id: str, recs_number: int):
        """
        Computes the predicted rating for specified user and items,
        should be used when a score prediction algorithm (instead of a ranking algorithm)
        was chosen in the config

        Args
            user_id: user for which calculate the predictions
            recs_number: how many items should the returned ranking contain,
            the ranking length can be lower
        Returns:
            (DataFrame): result frame whose columns are: to_id, rating

        Raises:
             ValueError: if the algorithm is a score prediction algorithm
        """
        if isinstance(self.__config.get_algorithm(), ScorePredictionAlgorithm):
            raise ValueError("You can't use rating prediction algorithms for this method")

        # load user ratings
        logger.info("Loading user ratings")
        user_ratings = self.__config.get_rating_frame()[self.__config.get_rating_frame()['from_id'].str.match(user_id)]

        # calculate predictions
        logger.info("Computing ranking")
        score_frame = self.__config.get_algorithm().predict(user_ratings, recs_number,
                                                            self.__config.get_items_directory())

        return score_frame

    def fit_eval(self, user_ratings: pd.DataFrame, test_set: pd.DataFrame):
        """
        Computes predicted ratings, or ranking (according to algorithm chosed in the config)
        user ratings will be used as train set to fit the algorithm.
        If the algorithm is score_prediction the rating for the item in the test set will
        be predicted, else a ranking with recs_number = len(test_set) will be computed

        Args
            user_ratings: train set
            test_set:
        Returns:
            (DataFrame): result frame whose columns are: to_id, rating
        """
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
