from abc import ABC, abstractmethod
from enum import Enum


class FieldContentProductionTechnique(ABC):
    """
    Abstract Class that manages to generalize the technique to use for producing one of the semantic description
    of an item's field's content.
    """
    def __init__(self):
        pass

    @abstractmethod
    def produce_content(self, field_data):
        """
        Abstract Method that produce content representation.
        Args:
            field_data: raw data, eventually preprocessed, from which obtain content representation

        Returns: instance of FieldContent

        """
        pass


class FieldToGraph(FieldContentProductionTechnique):
    """
    Abstract Class for techniques that use ontologies or LOD for producing the content representation.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def produce_content(self, field_data):
        pass


class EntityLinking(FieldContentProductionTechnique):
    """
    Abstract Class for implementations of entity linking techniques for producing the content representation.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def produce_content(self, field_data):
        pass


class Granularity(Enum):
    """
    Enumeration class whose elements are the possible units respect to which combine words for generating an embedding.
    """
    WORD = 1
    SENTENCE = 2
    DOC = 3


class CombiningTechnique(ABC):
    """
    Abstract Class that generalizes the modality in which imported embeddings will be
    combined to produce a content representation.
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
        for combining the embeddings
        embedding_source (EmbeddingSource): Source from which to get the embeddings
        granularity (Granularity): It can assume three values, depending on whether you want
        to combine words relatively to words, phrases or documents
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
            field_data: The embeddings to combine. This attribute can be of different kinds

        Returns:

        """
        pass
