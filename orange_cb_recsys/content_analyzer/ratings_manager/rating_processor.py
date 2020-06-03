from abc import ABC, abstractmethod


class RatingProcessor(ABC):
    ### COMMENTI

    @abstractmethod
    def fit(self, field_data: object):
        raise NotImplementedError


class SentimentalAnalysis(RatingProcessor):
    """
    Abstract Class that generalizes the sentimental analysis technique
    """

    @abstractmethod
    def fit(self, field_data: str):
        raise NotImplementedError


class NumberNormalizer(RatingProcessor):
    """
    Class that scale ratings in a numeric scale in the range [-1.0,1.0]
    """
    def __init__(self, field_name: str, min: float, max: float):
        self.__scale_factor = abs(max - min)

    def fit(self, field_data: float):
        return (field_data / self.__scale_factor) * 2 - 1
