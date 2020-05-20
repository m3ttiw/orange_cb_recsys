from abc import ABC, abstractmethod

from src.content_analyzer.information_processor.information_processor import InformationProcessor
from src.content_analyzer.memory_interfaces.memory_interfaces import InformationInterface


class EmbeddingLearner(ABC):
    """
    Abstract Class for the different kinds of embedding.

    Args:
        loader (InformationLoader): Instance of the class InformationLoader.
        preprocessor (InformationProcessor): Instance of the class InformationProcessor.
    """
    def __init__(self, loader: InformationInterface, preprocessor: InformationProcessor):
        self.__loader: InformationInterface = loader
        self.__preprocessor: InformationProcessor = preprocessor

    @abstractmethod
    def start_learning(self):
        raise NotImplementedError


class Word2Vec(EmbeddingLearner):
    """"
    Abstract Class for the different kinds of Word2Vec.
    """
    def __init__(self, loader: InformationInterface, preprocessor: InformationProcessor):
        super().__init__(loader, preprocessor)

    @abstractmethod
    def start_learning(self):
        """"
        Abstract Method
        """
        raise NotImplementedError


class Doc2Vec(EmbeddingLearner):
    """"
    Abstract Class for the different kinds of Doc2Vec.
    """
    def __init__(self, loader: InformationInterface, preprocessor: InformationProcessor):
        super().__init__(loader, preprocessor)

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
    def __init__(self, loader: InformationInterface, preprocessor: InformationProcessor):
        super().__init__(loader, preprocessor)

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
    def __init__(self, loader: InformationInterface, preprocessor: InformationProcessor):
        super().__init__(loader, preprocessor)

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
    def __init__(self, loader: InformationInterface, preprocessor: InformationProcessor):
        super().__init__(loader, preprocessor)

    @abstractmethod
    def start_learning(self):
        """"
        Abstract Method
        """
        raise NotImplementedError
