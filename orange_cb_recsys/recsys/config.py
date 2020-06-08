import os

from orange_cb_recsys.recsys.algorithm import RankingAlgorithm, Algorithm

from orange_cb_recsys.utils.const import home_path, DEVELOPING
from orange_cb_recsys.utils.load_ratings import load_ratings


class RecSysConfig:
    def __init__(self, users_directory: str,
                 items_directory: str,
                 algorithm: Algorithm,
                 rating_frame=None):
        if DEVELOPING:
            self.__users_directory: str = users_directory
            self.__items_directory: str = items_directory
        else:
            self.__users_directory: str = os.path.join(home_path, users_directory)
            self.__items_directory: str = os.path.join(home_path, items_directory)

        self.__algorithm: Algorithm = algorithm

        if type(rating_frame) is str:
            self.__rating_frame = load_ratings(rating_frame)
        else:
            self.__rating_frame = rating_frame

    def get_users_directory(self):
        return self.__users_directory

    def get_items_directory(self):
        return self.__items_directory

    def get_algorithm(self):
        return self.__algorithm

    def get_rating_frame(self):
        return self.__rating_frame

    def set_users_directory(self, users_directory: str):
        self.__users_directory = users_directory

    def set_ranking_algorithm(self, algorithm: str):
        self.__algorithm = algorithm

    def set_items_directory(self, items_directory: str):
        self.__items_directory = items_directory

    def set_rating_frame(self, rating_frame: str):
        self.__rating_frame = rating_frame
