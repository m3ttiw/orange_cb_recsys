from abc import ABC, abstractmethod
from enum import Enum


class FieldContentProductionTechnique(ABC):
    def __init__(self, primitive_content):
        self.__primitive_content = primitive_content

    def produce_description(self):
        pass


class FieldToGraph(FieldContentProductionTechnique, ABC):
    def __init__(self):
        super().__init__()


class EntityLinking(FieldContentProductionTechnique, ABC):
    def __init__(self):
        super().__init__()


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
