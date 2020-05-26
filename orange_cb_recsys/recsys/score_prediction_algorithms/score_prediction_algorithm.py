from abc import ABC, abstractmethod
from typing import List

from orange_cb_recsys.content_analyzer.content_representation.content import Content


class ScorePredictionAlgorithm(ABC):
    def __init__(self):
        pass


class RatingsSPA(ScorePredictionAlgorithm):
    def __init__(self, item_field: str):
        super().__init__()
        self.__item_field: str = item_field

    @abstractmethod
    def predict(self, user: Content, item: Content, ratings_field: str, items_directory: str) -> float:
        raise NotImplementedError


class UserInfoSPA(ScorePredictionAlgorithm):
    """
    Score prediction...
    """


class SingleFieldSPA(ScorePredictionAlgorithm):
    def __init__(self, user_field: str, item_field: str, user_field_representation: str = None, item_field_representation: str = None):
        super().__init__()
        self.__user_field: str = user_field
        self.__item_field: str = item_field
        self.__user_field_representation = user_field_representation
        self.__item_field_representation = item_field_representation

    def predict(self, user: Content, item: Content):
        raise NotImplementedError


class MultipleFieldSPA(ScorePredictionAlgorithm):
    def __init__(self, user_field: List[str], item_field: List[str]):
        super().__init__()
        self.__user_field: List[str] = user_field
        self.__item_field: List[str] = item_field

    def predict(self, user: Content, item: Content):
        raise NotImplementedError
