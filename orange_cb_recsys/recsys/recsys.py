import os
import pickle
from typing import List

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.recsys.ranking_algorithms.ranking_algorithm import RankingAlgorithm, TopNRanking
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import ScorePredictionAlgorithm, \
    RatingsSPA


class RecSys:
    def __init__(self, users_directory: str,
                 items_directory: str,
                 score_prediction_algorithm: ScorePredictionAlgorithm,
                 ranking_algorithm: RankingAlgorithm,
                 rating_field: str = None):
        self.__users_directory: str = users_directory
        self.__items_directory: str = items_directory
        self.__score_prediction_algorithm: ScorePredictionAlgorithm = score_prediction_algorithm
        self.__ranking_algorithm: RankingAlgorithm = ranking_algorithm
        self.__rating_field = rating_field

    def __get_item_filename_list(self) -> List[dir]:
        return os.listdir(self.__items_directory)

    def fit(self, user_filename: str, item_to_predict_filename_list: List[str] = None):
        user: Content = pickle.load(self.__users_directory + user_filename)
        if item_to_predict_filename_list is None:
            item_list = self.__get_item_filename_list()
            try:
                ratings_values = user.get_field(self.__rating_field).get_representation(str(0))
                item_to_predict_filename_list = [item for item in item_list if item not in ratings_values.get_value().keys()]  # lista di item non valutati
            except KeyError:
                item_to_predict_filename_list = item_list

        for item_filename in item_to_predict_filename_list:
            item = pickle.load(item_filename)

            if type(self.__score_prediction_algorithm) == RatingsSPA:
                assert (self.__rating_field is not None), "You must specify where to find ratings if you use ratings based algorithm"
                self.__score_prediction_algorithm.predict(user, item, self.__rating_field, self.__items_directory)
            else:
                self.__score_prediction_algorithm.predict(user, item)

        """
        score_dict = {}
        user: Content = pickle.load(self.__users_directory + user_filename)
        for item_filename in self.__get_item_filename_list():
            item: Content = pickle.load(item_filename)
            score_dict[item.get_content_id()] = self.__score_prediction_algorithm.predict(user, item)

        return self.__ranking_algorithm.rank(score_dict)
        """


# prova utilizzo
score_algorithm = RatingsSPA("item_field")
rank_algorithm = TopNRanking(5)
recsys = RecSys("users", "items", score_algorithm, rank_algorithm, "ratings")
