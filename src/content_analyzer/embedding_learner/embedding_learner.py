from abc import ABC, abstractmethod
from typing import List

from src.content_analyzer.information_processor.information_processor import InformationProcessor
from src.content_analyzer.raw_information_source import RawInformationSource


class EmbeddingLearner(ABC):
    """
    Abstract Class for the different kinds of embedding.

    Args:
        source (RawInformationSource): Source where the content is stored.
        preprocessor (InformationProcessor): Instance of the class InformationProcessor.
        field_list (List[str]): Field name list.
    """
    def __init__(self, source: RawInformationSource,
                 preprocessor: InformationProcessor,
                 field_list: List[str]):
        self.__source: RawInformationSource = source
        self.__preprocessor: InformationProcessor = preprocessor
        self.__field_list = field_list

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    def get_source(self):
        return self.__source

    def get_preprocessor(self):
        return self.__preprocessor

    def get_field_list(self):
        return self.__field_list
