import time
from abc import ABC, abstractmethod
from typing import List

from src.content_analyzer.information_processor.information_processor import TextProcessor
from src.content_analyzer.information_processor.nlp import NLTK
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
                 preprocessor: TextProcessor,
                 field_list: List[str]):
        self.__source: RawInformationSource = source
        if preprocessor is None:
            self.__preprocessor: TextProcessor = NLTK()
        else:
            self.__preprocessor: TextProcessor = preprocessor
        self.__field_list = field_list
        self.__model = None

    @abstractmethod
    def fit(self, **kwargs):
        raise NotImplementedError

    def get_source(self):
        return self.__source

    def get_preprocessor(self):
        return self.__preprocessor

    def get_field_list(self):
        return self.__field_list

    def set_model(self, model):
        self.__model = model

    def get_model(self):
        return self.__model

    def extract_corpus(self):
        corpus = []
        # iter the source
        for doc in self.get_source():
            doc_data = ""
            for field_name in self.get_field_list():
                # apply preprocessing and save the data in the list
                doc_data += " " + doc[field_name].lower()
            corpus.append(self.get_preprocessor().process(doc_data))
        return corpus

    def save(self):
        self.__model.save("../../embeddings/model_" + str(time.time()) + ".model")
