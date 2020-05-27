from abc import ABC, abstractmethod

from typing import Dict
import numpy as np


class FieldRepresentation(ABC):
    """
    Abstract class that generalize the concept of "field representation",
    a field representation is a semantic way to represent a field of an item.

    Args:
        name (str): name of the representation's instance
    """

    def __init__(self, name: str):
        self.__name = name

    def get_name(self) -> str:
        return self.__name

    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def get_value(self):
        raise NotImplementedError


class FeaturesBagField(FieldRepresentation):
    """
    Class for field representation using a bag of features,
    this class can be also used to represent a bag of words: <keyword, score>;
    this representation is produced by the EntityLinking and tf-idf techniques

    Args:
        features (dict<str, object>): the dictionary where features are stored
    """
    
    def __init__(self, name: str, features: Dict[str, object] = None):
        super().__init__(name)
        if features is None:
            features = {}
        self.__features: Dict[str, object] = features

    def __str__(self):
        representation_string = "Representation: " + self.get_name() + "\n"
        return representation_string + str(self.__features)

    def append_feature(self, feature_key: str, feature_value):
        """
        Add a feature (feature_key, feature_value) to the dict

        Args:
            feature_key (str): key, can be a url or a keyword
            feature_value: the value of the field

        """
        self.__features[feature_key] = feature_value

    def get_feature(self, feature_key):
        """
        Get the feature_value from the dict[feature_key]

        Args:
            feature_key (str): key, can be a url or a keyword

        Returns:
            feature_value: the value of the field
        """
        return self.__features[feature_key]

    def get_value(self) -> Dict[str, object]:
        """
        Get the features dict

        Returns:
            features: the features dict
        """
        return self.__features

    def __eq__(self, other):
        return self.__features == other.__features


class EmbeddingField(FieldRepresentation):
    """
    Class for field representation using embeddings(dense numeric vectors)
    this representation is produced by the EmbeddingTechnique.

    Examples:
        shape (4) = [x,x,x,x]
        shape (2,2) = [[x,x],
                       [x,x]]

    Args:
        embedding_array (np.ndarray): embeddings array,
            it can be of different shapes according to the granularity of the technique
    """
    def __init__(self, name: str,
                 embedding_array: np.ndarray):
        super().__init__(name)
        self.__embedding_array: np.ndarray = embedding_array

    def __str__(self):
        print("sto")
        representation_string = "Representation: " + self.get_name() + "\n\n"
        return representation_string + str(self.__embedding_array)

    def get_value(self) -> np.ndarray:
        return self.__embedding_array

    def __eq__(self, other):
        return self.__embedding_array == other.__embedding_array


class GraphField(FieldRepresentation):
    """
    Class for field representation using a graph.
    """


class ContentField:
    """
    Class that represents a field,
    a field can have different representation of itself
    Args:
        field_name (str): the name of the field
        timestamp (str): string that represents the timestamp
        representation_dict (list<FieldRepresentation>): the list of the representations.
    """

    def __init__(self, field_name: str,
                 timestamp: str = None,
                 representation_dict: Dict[str, FieldRepresentation] = None):
        if representation_dict is None:
            representation_dict = {}
        self.__timestamp = timestamp
        self.__field_name: str = field_name
        self.__representation_dict: Dict[str, FieldRepresentation] = representation_dict

    def __eq__(self, other) -> bool:
        """
        override of the method __eq__ of object class,

        Args:
            other (ContentField): the field to check if is equal to self

        Returns:
            bool: True if the names are equals
        """

        return self.__field_name == other.get_name() and self.__representation_dict == other.__representation_dict

    def __str__(self):
        field_string = "Field:" + self.__field_name
        rep_string = ""
        for rep in self.__representation_dict.values():
            rep_string += str(rep) + '\n\n'

        return field_string + '\n\n' + rep_string + "------------------------------"

    def append(self, representation_id: str, representation: FieldRepresentation):
        self.__representation_dict[representation_id] = representation

    def get_representation(self, representation_id: str):
        return self.__representation_dict[representation_id]

    def get_name(self) -> str:
        return self.__field_name