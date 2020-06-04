from abc import ABC, abstractmethod


class RatingProcessor(ABC):
    """
    Abstract class that process a rating with the personalized fit method
    that returns a score in the range [-1.0,1.0]
    """

    @abstractmethod
    def fit(self, field_data: object):
        raise NotImplementedError


class SentimentAnalysis(RatingProcessor):
    """
    Abstract Class that generalizes the sentiment analysis technique
    """

    @abstractmethod
    def fit(self, field_data: str):
        raise NotImplementedError


class NumberNormalizer(RatingProcessor):
    """
    Class that scale ratings in a numeric scale in the range [-1.0,1.0]
    """
    def __init__(self, min_: float, max_: float):
        if min_ > max_:
            max_, min_ = min_, max_
        self.__min = min_
        self.__max = max_
        self.__scale_span = abs(min_ - max_)

    def fit(self, field_data: float):
        if field_data < self.__min:
            return self.__min
        if field_data > self.__max:
            return self.__max
        return (float(field_data - self.__min) / float(self.__scale_span)) * 2 - 1
