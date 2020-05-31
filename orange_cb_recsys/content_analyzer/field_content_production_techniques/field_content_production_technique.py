from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import nltk
import numpy as np

from nltk.tokenize import sent_tokenize
from orange_cb_recsys.content_analyzer.content_representation.content_field import FieldRepresentation, \
    FeaturesBagField, EmbeddingField, GraphField
from orange_cb_recsys.content_analyzer.information_processor.information_processor import InformationProcessor
from orange_cb_recsys.content_analyzer.memory_interfaces.text_interface import IndexInterface
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.utils.check_tokenization import check_tokenized


class FieldContentProductionTechnique(ABC):
    """
    Abstract class that generalizes the techniques to use for producing the semantic description
    of a content's field's representation
    """

    def __init__(self):
        pass


class SentimentalAnalysis(FieldContentProductionTechnique):
    """
    Abstract Class that generalizes the sentimental analysis technique
    """

    def __init__(self, field_name: str, source: RawInformationSource):
        super().__init__()
        self.__field_name = field_name
        self.__source = source

    @abstractmethod
    def calculate_score(self):
        pass

    def get_source(self):
        return self.__source

    def get_field_name(self):
        return self.__field_name


class CollectionBasedTechnique(FieldContentProductionTechnique):
    """
    This class generalizes the techniques that work on the entire content collection, like the tf-idf technique
    """

    def __init__(self):
        super().__init__()
        self.__need_refactor: Dict[Tuple[str, str], List[InformationProcessor]] = {}

    def append_field_need_refactor(self, field_name: str, pipeline_id, processor_list: List[InformationProcessor]):
        self.__need_refactor[(field_name, pipeline_id)] = processor_list

    def get_need_refactor(self):
        return self.__need_refactor

    @abstractmethod
    def produce_content(self, field_representation_name: str, content_id: str,
                        field_name: str, pipeline_id: str) -> FieldRepresentation:
        raise NotImplementedError

    @abstractmethod
    def dataset_refactor(self, information_source: RawInformationSource, id_field_names):
        raise NotImplementedError

    def __str__(self):
        return "CollectionBasedTechnique"

    def __repr__(self):
        return "CollectionBasedTechnique " + str(self.__need_refactor)


class SingleContentTechnique(FieldContentProductionTechnique):
    @abstractmethod
    def produce_content(self, field_representation_name: str, field_data) -> FieldRepresentation:
        """
        Given data of certain field it returns a complex representation's instance of the field.
        Args:
            field_representation_name: name of the field representation object that will be created
            field_data: input for the complex representation production

        Returns:
            FieldRepresentation: an instance of FieldRepresentation,
                 the particular type of representation depends from the technique
        """


class FieldToGraph(SingleContentTechnique):
    """
    Abstract class that generalizes techniques
    that use ontologies or LOD for producing the semantic description
    """

    @abstractmethod
    def produce_content(self, field_representation_name: str, field_data: str) -> GraphField:
        raise NotImplementedError

    def __str__(self):
        return "FieldToGraph"

    def __repr__(self):
        return "FieldToGraph " + "graph content"


class TfIdfTechnique(CollectionBasedTechnique):
    """
    Class that produce a Bag of words with tf-idf metric
    """

    def __init__(self):
        super().__init__()
        self.__index = IndexInterface('./frequency-index')

    @abstractmethod
    def produce_content(self, field_representation_name: str, content_id: str,
                        field_name: str, pipeline_id: str) -> FeaturesBagField:
        raise NotImplementedError

    @abstractmethod
    def dataset_refactor(self, information_source: RawInformationSource, id_field_names: str):
        raise NotImplementedError

    def __str__(self):
        return "TfIdfTechnique"

    def __repr__(self):
        return "TfIdfTechnique " + str(self.__index)


class EntityLinking(SingleContentTechnique):
    """
    Abstract class that generalizes implementations that use entity linking
    for producing the semantic description
    """

    @abstractmethod
    def produce_content(self, field_representation_name: str, field_data) -> FeaturesBagField:
        raise NotImplementedError


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
        Combine, in a way specified in the implementations,
        the row of the input matrix

        Args:
            embedding_matrix: matrix whose rows will be combined

        Returns:

        """
        raise NotImplementedError


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

    def load(self, text: List[str]) -> np.ndarray:
        """
        Function that extracts from the embeddings model
        the vectors of the words contained in text

        Args:
            text (str): contains words of which vectors will be extracted

        Returns:
            np.ndarray: bi-dimensional numpy vector,
                each row is a term vector
        """
        embedding_matrix = np.ndarray(shape=(len(text), self.get_vector_size()))

        text = check_tokenized(text)

        for i, word in enumerate(text):
            word = word.lower()
            try:
                embedding_matrix[i, :] = self.__model[word]
            except KeyError:
                embedding_matrix[i, :] = np.zeros(self.get_vector_size())

        return embedding_matrix

    def set_model(self, model):
        self.__model = model

    def get_vector_size(self) -> int:
        return self.__model.vector_size

    def get_model(self):
        return self.__model

    def __str__(self):
        return "EmbeddingSource"

    def __repr__(self):
        return "EmbeddingSource " + str(self.__model)


class EmbeddingTechnique(SingleContentTechnique):
    """
    Class that can be used to combine different embeddings coming to various sources
    in order to produce the semantic description.

    Args:
        combining_technique (CombiningTechnique): The technique that will be used
        for combining the embeddings.
        embedding_source (EmbeddingSource):
            Source from which extract the embeddings vectors for the words in field_data.
        granularity (Granularity): It can assume three values,
            depending on whether framework user want
            to combine relatively to words, phrases or documents.
    """

    def __init__(self, combining_technique: CombiningTechnique,
                 embedding_source: EmbeddingSource,
                 granularity: str):
        super().__init__()
        self.__combining_technique: CombiningTechnique = combining_technique
        self.__embedding_source: EmbeddingSource = embedding_source

        self.__granularity: str = granularity.lower()

    def produce_content(self, field_representation_name: str, field_data) -> EmbeddingField:
        """
        Method that builds the semantic content starting from the embeddings contained in
        field_data.
        Args:
            field_representation_name:
            field_data: The terms whose embeddings will be combined.

        Returns:
            np.ndarray:
                mono-dimensional array for DOC embedding
                bi-dimensional array for SENTENCE and WORD embedding
        """

        if self.__granularity == "word":
            doc_matrix = self.__embedding_source.load(field_data)
            return EmbeddingField(field_representation_name, doc_matrix)
        elif self.__granularity == "sentence":
            try:
                nltk.data.find('punkt')
            except LookupError:
                nltk.download('punkt')

            sentences = sent_tokenize(field_data)
            for i, sentence in enumerate(sentences):
                sentences[i] = sentence[:len(sentence) - 1]

            sentences_embeddings = np.ndarray(shape=(len(sentences), self.__embedding_source.get_vector_size()))
            for i, sentence in enumerate(sentences):
                sentence_matrix = self.__embedding_source.load(sentence)
                sentences_embeddings[i, :] = self.__combining_technique.combine(sentence_matrix)

            return EmbeddingField(field_representation_name, sentences_embeddings)
        elif self.__granularity == "doc":
            doc_matrix = self.__embedding_source.load(field_data)
            return EmbeddingField(field_representation_name, self.__combining_technique.combine(doc_matrix))
        else:
            raise ValueError("Must specify a valid embedding technique granularity")

    def __str__(self):
        return "EmbeddingTechnique"

    def __repr__(self):
        return "EmbeddingTechnique " + str(self.__combining_technique) + " " + str(self.__embedding_source) + " " + str(
            self.__granularity)
