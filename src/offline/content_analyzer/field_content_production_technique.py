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


class EmbeddingTechnique(FieldContentProductionTechnique):
    def __init__(self, combining_technique: CombiningTechnique,
                 embedding_source: EmbeddingSource,
                 granularity: Granularity):
        super().__init__()
        self.__combining_technique: CombiningTechnique = combining_technique
        self.__embedding_source: EmbeddingSource = embedding_source
        self.__granularity: Granularity = granularity

    def produce_content(self, field_data):
        pass
