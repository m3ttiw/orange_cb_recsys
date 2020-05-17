import re
from abc import ABC, abstractmethod


class InformationProcessor(ABC):
    """
    Abstract class that generalize the data processing.
    """
    @abstractmethod
    def process(self, field_data):
        raise NotImplementedError


class ImageProcessor(InformationProcessor):
    """
    Abstract class for image processing.
    """
    @abstractmethod
    def process(self, field_data):
        raise NotImplementedError


class AudioProcessor(InformationProcessor):
    """
    Abstract class for audio processing.
    """
    @abstractmethod
    def process(self, field_data):
        raise NotImplementedError


class TextProcessor(InformationProcessor):
    """
    Abstract class for raw text processing.
    """
    @abstractmethod
    def process(self, field_data):
        raise NotImplementedError


class NLP(TextProcessor):
    """
    Class for processing a text via Natural Language Processing.

    Args:
        stopwords_removal (bool): Whether you want to remove stop words
        stemming (bool): Whether you want to perform stemming
        lemmatization (bool):  Whether you want to perform lemmatization
        strip_multiple_whitespaces (bool): Whether you want to remove multiple whitespaces
        url_tagging (bool): Whether you want to tag the urls in the text and to replace with "<URL>"
    """

    def __init__(self, stopwords_removal: bool = False,
                 stemming: bool = False,
                 lemmatization: bool = False,
                 strip_multiple_whitespaces: bool = True,
                 url_tagging: bool = False,
                 named_entity_recognition: bool = False):
        super().__init__()
        self.__stopwords_removal: bool = stopwords_removal
        self.__stemming: bool = stemming
        self.__lemmatization: bool = lemmatization
        self.__strip_multiple_whitespaces: bool = strip_multiple_whitespaces
        self.__url_tagging: bool = url_tagging
        self.__is_tokenized = False
        self.__named_entity_recognition: bool = named_entity_recognition

    def get_is_tokenized(self):
        return self.__is_tokenized

    def get_stopwords_removal(self):
        return self.__stopwords_removal

    def get_stemming(self):
        return self.__stemming

    def get_lemmatization(self):
        return self.__lemmatization

    def get_strip_multiple_whitespaces(self):
        return self.__strip_multiple_whitespaces

    def get_url_tagging(self):
        return self.__url_tagging

    def get_named_entity_recognition(self):
        return self.__named_entity_recognition

    def set_stopwords_removal(self, stopwords_removal):
        self.__stopwords_removal = stopwords_removal

    def set_stemming(self, stemming):
        self.__stemming = stemming

    def set_lemmatization(self, lemmatization):
        self.__lemmatization = lemmatization

    def set_strip_multiple_whitespaces(self, strip_multiple_whitespaces):
        self.__strip_multiple_whitespaces = strip_multiple_whitespaces

    def set_url_tagging(self, url_tagging):
        self.__url_tagging = url_tagging

    def set_is_tokenized(self, is_tokenized: bool):
        self.__is_tokenized = is_tokenized

    def set_named_entity_recognition(self, named_entity_recognition: bool):
        self.__named_entity_recognition = named_entity_recognition

    @abstractmethod
    def process(self, field_data) -> str:
        """
        Apply on original text operations with True value
        Args:
            field_data: text on which NLP with specified phases will be applied

        Returns:
            list<str>: The text, after being processed with the specified NLP pipeline,
                is splitted in single words that are put into a list
        """
        raise NotImplementedError

