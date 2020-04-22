from abc import ABC
from enum import Enum


class ItemContentProductionTechnique(ABC):
    def __init__(self):
        pass

    def produce_description(self):
        pass


class ItemToGraph(ItemContentProductionTechnique, ABC):
    def __init__(self):
        super().__init__()


class EntityLinking(ItemContentProductionTechnique, ABC):
    def __init__(self):
        super().__init__()


class Granularity(Enum):
    WORD = 1
    SENTENCE = 2
    DOC = 3


class EmbeddingTechnique(ItemContentProductionTechnique):
    def __init__(self, combining_technique, embedding_source, granularity: Granularity):
        super().__init__()
        self.__combining_technique = combining_technique
        self.__embedding_source = embedding_source
        self.__granularity = granularity
