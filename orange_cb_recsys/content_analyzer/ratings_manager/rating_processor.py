from abc import ABC, abstractmethod


class RatingProcessor(ABC):
    ### COMMENTI
    def __init__(self, field_name: str):
        self.__field_name = field_name

    def get_field_name(self):
        return self.__field_name

    @abstractmethod
    def __type_check(self, field_data: object):
        raise NotImplementedError

    @abstractmethod
    def fit(self, field_data: object):
        raise NotImplementedError


class SentimentalAnalysis(RatingProcessor):
    ### COMMENTI
    """
    Abstract Class that generalizes the sentimental analysis technique
    """

    def __init__(self, field_name: str):
        super().__init__(field_name)

    def __type_check(self, field_data: object):
        if type(field_data) is not str:
            raise TypeError("Sentiment Analisys works only in textual fields")

    @abstractmethod
    def fit(self, field_data: object):
        raise NotImplementedError
