from abc import ABC, abstractmethod
from enum import Enum


class FieldContentProductionTechnique(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def produce_content(self, field_data):
        pass


class FieldToGraph(FieldContentProductionTechnique):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def produce_content(self, field_data):
        pass


class EntityLinking(FieldContentProductionTechnique):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def produce_content(self, field_data):
        pass


class Granularity(Enum):
    WORD = 1
    SENTENCE = 2
    DOC = 3


class EmbeddingTechnique(FieldContentProductionTechnique):
    def __init__(self, combining_technique, embedding_source, granularity: Granularity):
        super().__init__()
        self.__combining_technique = combining_technique
        self.__embedding_source = embedding_source
        self.__granularity = granularity

    def produce_content(self, field_data):
        pass


class CombiningTechnique(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def combine(self):
        pass


class EmbeddingSource(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load(self):
        pass
