from abc import ABC

from typing import List, Dict
import numpy as np


class FieldRepresentation(ABC):
    """
    Abstract class that generalize the concept of "field representation",
    a field representation is a semantic way to represent a field of an item.
    """
    def __init__(self, name: str):
        self.__name = name

    def get_name(self) -> str:
        return self.__name


class FeaturesBagField(FieldRepresentation):
    """
    Class for field representation using a bag of features,
    this class can be also used to represent a bag of words: <keyword, score>;
    this representation is produced by the EntityLinking and tf-idf techniques

    Args:
        features (dict): <str, object> the dictionary where features are stored
    """
    def __init__(self, name: str, features: Dict[str, object] = None):
        super().__init__(name)
        if features is None:
            features = {}
        self.__features: Dict[str, object] = features

    def append_feature(self, feature_key: str, feature_value):
        pass


class EmbeddingField(FieldRepresentation):
    """
    Class for field representation using embeddings(dense numeric vectors)
    this representation is produced by the EmbeddingTechnique.

    Examples:
        shape (4) = [x,x,x,x]
        shape (2,2) = [[x,x],
                       [x,x]]

    Args:
        embedding_array: embeddings array,
        it can be of different shapes according to the granularity of the technique
    """
    def __init__(self, name: str, embedding_array: np.ndarray):
        super().__init__(name)
        self.__embedding_array: np.ndarray = embedding_array


class GraphField(FieldRepresentation):
    """
    Class for field representation using a graph.
    """
    def __init__(self, name: str):
        super().__init__(name)


class ContentField:
    """
    Class that represents a field,
    a field can have different representation of itself
    Args:
        field_name (str): the name of the field
        representations (list<FieldRepresentation>): the list of the representations.
    """
    def __init__(self, field_name: str, representations: List[FieldRepresentation] = None):
        if representations is None:
            representations = []
        self.__field_name: str = field_name
        self.__representations: List[FieldRepresentation] = representations

    def __eq__(self, other):
        """
        override of the method __eq__ of object class,

        Args:
            other (ContentField): the field to check if is equal to self

        Returns:
            True if the names are the same
        """
        return self.__field_name == other.get_name()

    def append(self, representation: FieldRepresentation):
        self.__representations.append(representation)

    def get_name(self) -> str:
        return self.__field_name
