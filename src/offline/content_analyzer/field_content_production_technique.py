from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from offline.content_analyzer.content_representation.content_field import EmbeddingField
from src.offline.memory_interfaces.memory_interfaces import InformationInterface


class FieldContentProductionTechnique(ABC):
    """
    Abstract class that manages to generalize the technique to use for producing the semantic description
    of an item's field's content
    """
    def __init__(self):
        pass

    @abstractmethod
    def produce_content(self, field_representation_name: str, field_data):
        pass


class FieldToGraph(FieldContentProductionTechnique):
    """
    Class that uses ontologies or LOD for producing the semantic description
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def produce_content(self, field_representation_name: str, field_data):
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

    def produce_content(self, field_representation_name: str, field_data):
        print("Creating bag of words")


class EntityLinking(FieldContentProductionTechnique):
    """
    Class that uses entity linking techniques for producing the semantic description
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def produce_content(self, field_representation_name: str, field_data):
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
    def combine(self, embedding_matrix: np.ndarray):
        pass


class EmbeddingSource(ABC):
    """
    General class whose purpose is to load the previously learned embeddings to combine.
    """
    def __init__(self):
        self.__model = None

    def load(self, text: str):
        """
        Function that loads the embeddings from the file.

        Returns:
            The loaded embedding matrix
        """
        words = text.split(" ")
        embedding_matrix = np.ndarray(shape=(len(words), self.get_vector_size()))

        for i, word in enumerate(words):
            embedding_matrix[i, :] = self.__model[word]

        return embedding_matrix

    def set_model(self, model):
        self.__model = model

    def get_vector_size(self):
        return self.__model.vector_size


class SentenceDetectionTechnique(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def detect_sentences(self, text: str):
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
                 sentence_detection: SentenceDetectionTechnique,
                 granularity: Granularity):
        super().__init__()
        self.__combining_technique: CombiningTechnique = combining_technique
        self.__embedding_source: EmbeddingSource = embedding_source
        self.__sentence_detection: SentenceDetectionTechnique = sentence_detection
        self.__granularity: Granularity = granularity

    def produce_content(self, field_representation_name: str, field_data):
        """
        Method that builds the semantic content starting from the embeddings contained in
        field_data.
        Args:
            field_representation_name:
            field_data: The terms whose embeddings will be combined.

        Returns:

        """

        if self.__granularity == Granularity.WORD:
            doc_matrix = self.__embedding_source.load(field_data)
            return EmbeddingField("Embedding", doc_matrix)
        elif self.__granularity == Granularity.SENTENCE:
            sentences = self.__sentence_detection.detect_sentences(field_data)
            sentences_embeddings = np.ndarray(shape=(len(sentences), self.__embedding_source.get_vector_size()))
            for i, sentence in enumerate(sentences):
                sentence_matrix = self.__embedding_source.load(sentence)
                sentences_embeddings[i, :] = self.__combining_technique.combine(sentence_matrix)

            return sentences_embeddings
        elif self.__granularity == Granularity.DOC:
            doc_matrix = self.__embedding_source.load(field_data)
            return self.__combining_technique.combine(doc_matrix)
