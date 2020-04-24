from abc import ABC, abstractmethod

from src.offline.content_analyzer.information_loader import InformationLoader
from src.offline.content_analyzer.information_processor import InformationProcessor


class EmbeddingLearner(ABC):
    """
    Abstract Class for the different kinds of embedding.

    Args:
        loader (InformationLoader): instance of the class InformationLoader
        preprocessor (InformationProcessor): instance of the class InformationProcessor
    """
    def __init__(self, loader: InformationLoader, preprocessor: InformationProcessor):
        self.__loader: InformationLoader = loader
        self.__preprocessor: InformationProcessor = preprocessor

    @abstractmethod
    def start_learning(self):
        """"
        Abstract method implemented in the different kinds of embedding to start learning.
        """
        pass


class Word2Vec(EmbeddingLearner):
    """"
    Abstract Class for the different implementations of Word2Vec.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def start_learning(self):
        pass


class LatentSemanticAnalysis(EmbeddingLearner):
    """"
    Abstract Class for the different implementations of Latent Semantic Analysis.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def start_learning(self):
        pass


class RandomIndexing(EmbeddingLearner):
    """"
    Abstract Class for the different implementations of Random Indexing.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def start_learning(self):
        pass


class ExplicitSemanticAnalysis(EmbeddingLearner):
    """"
    Abstract Class for the different implementations of Explicit Semantic Analysis.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def start_learning(self):
        pass
