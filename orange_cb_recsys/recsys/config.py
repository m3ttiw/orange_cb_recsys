import os

from orange_cb_recsys.recsys.ranking_algorithms.ranking_algorithm import RankingAlgorithm
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import ScorePredictionAlgorithm

from orange_cb_recsys.utils.const import home_path, DEVELOPING
from orange_cb_recsys.utils.load_ratings import load_ratings


class RecSysConfig:
    def __init__(self, users_directory: str,
                 items_directory: str,
                 score_prediction_algorithm: ScorePredictionAlgorithm,
                 ranking_algorithm: RankingAlgorithm,
                 rating_frame=None):
        if DEVELOPING:
            self.__users_directory: str = users_directory
            self.__items_directory: str = items_directory
        else:
            self.__users_directory: str = os.path.join(home_path, users_directory)
            self.__items_directory: str = os.path.join(home_path, items_directory)

        self.__score_prediction_algorithm: ScorePredictionAlgorithm = score_prediction_algorithm
        self.__ranking_algorithm: RankingAlgorithm = ranking_algorithm
        if type(rating_frame) is str:
            self.__rating_frame = load_ratings(rating_frame)
        else:
            self.__rating_frame = rating_frame

    def get_users_directory(self):
        return self.__users_directory

    def get_items_directory(self):
        return self.__items_directory

    def get_score_prediction_algorithm(self):
        return self.__score_prediction_algorithm

    def get_ranking_algorithm(self):
        return self.__ranking_algorithm

    def get_rating_frame(self):
        return self.__rating_frame

    def set_users_directory(self, users_directory: str):
        self.__users_directory = users_directory

    def set_ranking_algorithm(self, ranking_algorithm: str):
        self.__ranking_algorithm = ranking_algorithm

    def set_items_directory(self, items_directory: str):
        self.__items_directory = items_directory

    def set_score_prediction_algorithm(self, score_prediction_algorithm: str):
        self.__score_prediction_algorithm = score_prediction_algorithm

    def set_rating_frame(self, rating_frame: str):
        self.__rating_frame = rating_frame
