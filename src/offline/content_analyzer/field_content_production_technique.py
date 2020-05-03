from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from offline.memory_interfaces.memory_interfaces import InformationInterface


class FieldContentProductionTechnique(ABC):
    """
    Abstract class that manages to generalize the technique to use for producing the semantic description
    of an item's field's content
    """
    def __init__(self):
        pass

    @abstractmethod
    def produce_content(self, field_data):
        pass


class FieldToGraph(FieldContentProductionTechnique):
    """
    Class that uses ontologies or LOD for producing the semantic description
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def produce_content(self, field_data):
        pass


class TfIdfTechnique(FieldContentProductionTechnique):
    """
    Class that produce a Bag of word with tf-idf metric
    Args:
        memory_interface (InformationInterface): the memory interface for managing the data stored
    """
    def __init__(self, memory_interface: InformationInterface):
        self.__memory_interface = memory_interface
        super().__init__()

    def produce_content(self, field_data):
        print("Creating bag of words")


class EntityLinking(FieldContentProductionTechnique):
    """
    Class that uses entity linking techniques for producing the semantic description
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def produce_content(self, field_data):
        pass


class Granularity(Enum):
    """
    Enumeration class whose elements are the possible units respect to which combine for generating an embedding.
    """
    WORD = 1
    SENTENCE = 2
    DOC = 3


class CombiningTechnique(ABC):
    """
    Class that generalizes the modality in which previously learned embeddings will be
    combined to produce a semantic description.
    """
    def __init__(self):
        pass

    @abstractmethod
    def combine(self):
        pass


class EmbeddingSource(ABC):
    """
    General class whose purpose is to load the previously learned embeddings to combine.
    """
    def __init__(self):
        pass

    @abstractmethod
    def load(self):
        pass


class EmbeddingTechnique(FieldContentProductionTechnique):
    """
    Class that can be used to combine different embeddings coming to various sources
    in order to produce the semantic description.

    Attributes:
        combining_technique (CombiningTechnique): The technique that will be used
        for combining the embeddings.
        embedding_source (EmbeddingSource): Source from which to get the embeddings.
        granularity (Granularity): It can assume three values, depending on whether you want
        to combine relatively to words, phrases or documents.
    """
    def __init__(self, combining_technique: CombiningTechnique,
                 embedding_source: EmbeddingSource,
                 granularity: Granularity):
        super().__init__()
        self.__combining_technique: CombiningTechnique = combining_technique
        self.__embedding_source: EmbeddingSource = embedding_source
        self.__granularity: Granularity = granularity

    def produce_content(self, field_data):
        """
        Method that builds the semantic content starting from the embeddings contained in
        field_data.
        Args:
            field_data: The terms whose embeddings will be combined.

        Returns:

        """

        print("Creating matrix")
