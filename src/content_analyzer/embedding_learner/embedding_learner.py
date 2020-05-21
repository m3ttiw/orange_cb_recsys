from abc import ABC, abstractmethod

from src.content_analyzer.information_processor.information_processor import InformationProcessor
from src.content_analyzer.raw_information_source import RawInformationSource


class EmbeddingLearner(ABC):
    """
    Abstract Class for the different kinds of embedding.

    Args:
        source (RawInformationSource): Source where the content is stored.
        preprocessor (InformationProcessor): Instance of the class InformationProcessor.
    """
    def __init__(self, source: RawInformationSource, preprocessor: InformationProcessor):
        self.__source: RawInformationSource = source
        self.__preprocessor: InformationProcessor = preprocessor

    @abstractmethod
    def start_learning(self):
        raise NotImplementedError


class Word2Vec(EmbeddingLearner):
    """"
    Abstract Class for the different kinds of Word2Vec.
    """
    def __init__(self, source: RawInformationSource, preprocessor: InformationProcessor):
        super().__init__(source, preprocessor)

    @abstractmethod
    def start_learning(self):
        """"
        Abstract Method
        """
        raise NotImplementedError


class LatentSemanticAnalysis(EmbeddingLearner):
    """"
    Abstract Class for the different kinds of Latent Semantic Analysis.
    """
    def __init__(self, source: RawInformationSource, preprocessor: InformationProcessor):
        super().__init__(source, preprocessor)

    @abstractmethod
    def start_learning(self):
        """"
        Abstract Method
        """
        raise NotImplementedError


class RandomIndexing(EmbeddingLearner):
    """"
    Abstract Class for the different kinds of Random Indexing.
    """
    def __init__(self, source: RawInformationSource, preprocessor: InformationProcessor):
        super().__init__(source, preprocessor)

    @abstractmethod
    def start_learning(self):
        """"
        Abstract Method
        """
        raise NotImplementedError


class ExplicitSemanticAnalysis(EmbeddingLearner):
    """"
    Abstract Class for the different kinds of Explicit Semantic Analysis.
    """
    def __init__(self, source: RawInformationSource, preprocessor: InformationProcessor):
        super().__init__(source, preprocessor)

    @abstractmethod
    def start_learning(self):
        """"
        Abstract Method
        """
        raise NotImplementedError
