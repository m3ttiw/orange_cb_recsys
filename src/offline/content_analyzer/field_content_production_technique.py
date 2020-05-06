from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import numpy as np
from src.offline.content_analyzer.content_representation.content_field \
    import EmbeddingField, FeaturesBagField, FieldRepresentation, GraphField
from src.offline.memory_interfaces.memory_interfaces import InformationInterface


class FieldContentProductionTechnique(ABC):
    """
    Abstract class that generalize the technique to use for producing the semantic description
    of a content's field's representation
    """
    def __init__(self):
        pass

    @abstractmethod
    def produce_content(self, field_data) -> FieldRepresentation:
        """

        Args:
            field_data: input for the complex representation production

        Returns:
            FieldRepresentation: an instance of FieldRepresentation,
                 the particular type of representation depends from the technique
        """
        pass


class FieldToGraph(FieldContentProductionTechnique):
    """
    Abstract class that generalize techniques
    that uses ontologies or LOD for producing the semantic description
    """
    @abstractmethod
    def produce_content(self, field_data) -> GraphField:
        pass


class TfIdfTechnique(FieldContentProductionTechnique):
    """
    Class that produce a Bag of words with tf-idf metric
    Args:
        memory_interface (InformationInterface):
            the memory interface where all of the collection's terms are stored,
            useful for frequency calc
    """
    def __init__(self, memory_interface: InformationInterface):
        super().__init__()
        self.__memory_interface = memory_interface

    def produce_content(self, field_data) -> FeaturesBagField:
        print("Creating bag of words")


class EntityLinking(FieldContentProductionTechnique):
    """
    Abstract class that generalize implementations
    that uses entity linking for producing the semantic description
    """
    @abstractmethod
    def produce_content(self, field_data) -> FeaturesBagField:
        pass


class Granularity(Enum):
    """
    Enumeration whose elements are the possible units
    respect to which combine for generating an embedding.
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
        """
        Combine ,in a way specified in the implementations,
        the row of the input matrix

        Args:
            embedding_matrix: matrix whose rows will be combined

        Returns:

        """
        pass


class EmbeddingSource(ABC):
    """
    General class whose purpose is to
    store the loaded pre-trained embeddings model and
    extract from it specified words

    Args:
        self.__model: embeddings model loaded from source
    """
    def __init__(self):
        self.__model = None

    def load(self, text: str) -> np.ndarray:
        """
        Function that extract from the embeddings model
        the vectors of thw words contained in text

        Args:
            text (str): contains words of which vectors will be extracted

        Returns:
            np.ndarray: bi-dimensional numpy vector,
                each row is a term vector
        """
        words = text.split(" ")
        embedding_matrix = np.ndarray(shape=(len(words), self.get_vector_size()))

        for i, word in enumerate(words):
            embedding_matrix[i, :] = self.__model[word]

        return embedding_matrix

    def set_model(self, model):
        self.__model = model

    def get_vector_size(self) -> int:
        return self.__model.vector_size


class SentenceDetectionTechnique(ABC):
    """
    Abstract class that generalizes implementation of
    techniques used to divide a text in sentences
    """
    def __init__(self):
        pass

    @abstractmethod
    def detect_sentences(self, text: str) -> List[str]:
        """
        Divide the input text in a list of sentences

        Args:
            text (str): text that will be divided

        Returns:
            List<str>: list of sentences
        """
        pass


class EmbeddingTechnique(FieldContentProductionTechnique):
    """
    Class that can be used to combine different embeddings coming to various sources
    in order to produce the semantic description.

    Args:
        combining_technique (CombiningTechnique): The technique that will be used
        for combining the embeddings.
        embedding_source (EmbeddingSource):
            Source from which extract the embeddings vectors for the words in field_data.
        sentence_detection (SentenceDetectionTechnique): technique that wil lbe use to divide
            the text in sentences if the granularity specified is SENTENCE
        granularity (Granularity): It can assume three values,
            depending on whether framework user want
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

    def produce_content(self, field_data) -> np.ndarray:
        """
        Method that builds the semantic content starting from the embeddings contained in
        field_data.
        Args:
            field_data: The terms whose embeddings will be combined.

        Returns:
            np.ndarray:
                mono-dimensional array for DOC embedding
                bi-dimensional array for SENTENCE and WORD embedding
        """

        if self.__granularity == Granularity.WORD:
            doc_matrix = self.__embedding_source.load(field_data)
            return EmbeddingField("Embedding", doc_matrix)
        if self.__granularity == Granularity.SENTENCE:
            sentences = self.__sentence_detection.detect_sentences(field_data)
            sentences_embeddings = np.ndarray(shape=(len(sentences),
                                                     self.__embedding_source.get_vector_size()))
            for i, sentence in enumerate(sentences):
                sentence_matrix = self.__embedding_source.load(sentence)
                sentences_embeddings[i, :] = self.__combining_technique.combine(sentence_matrix)

            return sentences_embeddings
        if self.__granularity == Granularity.DOC:
            doc_matrix = self.__embedding_source.load(field_data)
            return self.__combining_technique.combine(doc_matrix)
