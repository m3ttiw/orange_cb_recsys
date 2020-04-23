from abc import ABC, abstractmethod

from src.offline.content_analyzer.information_loader import InformationLoader
from src.offline.content_analyzer.information_processor import InformationProcessor


class EmbeddingLearner(ABC):
    def __init__(self, loader: InformationLoader, preprocessor: InformationProcessor):
        self.__loader: InformationLoader = loader
        self.__preprocessor: InformationProcessor = preprocessor

    @abstractmethod
    def start_learning(self):
        pass


class Word2Vec():
    def __init__(self):
        super().__init__()

    @abstractmethod
    def start_learning(self):
        pass


class LatentSemanticAnalysis():
    def __init__(self):
        super().__init__()

    @abstractmethod
    def start_learning(self):
        pass


class RandomIndexing():
    def __init__(self):
        super().__init__()

    @abstractmethod
    def start_learning(self):
        pass


class ExplicitSemanticAnalysis():
    def __init__(self):
        super().__init__()

    @abstractmethod
    def start_learning(self):
        pass
