from abc import ABC
import numpy as np


class FieldContent(ABC):
    """
    Abstract class that generalize the concept of "field representation".
    A field representation is a semantic way to represent a field of an item.
    """
    def __init__(self):
        pass


class FeaturesBagField(FieldContent):
    """
    Class for represent a baf of feature.
    This class can represent a bag of word if the key value of the dict "features" is a keyword
    instead of a url for represent a bag of feature.

    Args:
        features (dict): <str, object> the dictionary where feature are indexed
    """
    def __init__(self, features: dict[str, object] = None):
        super().__init__()
        if features is None:
            features = {}
        self.__features: dict = features

    def add_feature(self, feature_key: str, feature_value):
        """
        Add a feature (feature_key, feature_value) to the dict

        Args:
            feature_key (str): key, can be a url or a keyword
            feature_value: the value of the field

        Returns:

        """
        pass


class EmbeddingField(FieldContent):
    """
    Class for represent a embedding.
    A embedding is a dense numeric vector.
    The shape of the array can be set for a n dimensional array, with n > 1

    Examples:
        shape (4) = [x,x,x,x]
        shape (2,2) = [[x,x],
                       [x,x]]

    Args:
        shape (tuple): is the shape of the array
    """
    def __init__(self, shape: tuple):
        super().__init__()
        self.__shape: tuple = shape
        self.__embedding_array = np.ndarray(shape=self.__shape)

    def add_value(self, value: float, coords: tuple):
        """
        Add the value in the array at coords. Coords need to be coherent to the shape of the array.

        Raises:
            general Exception("len(coords) != len(self.__shape)")

        Args:
            value (float): the value to be added
            coords (tuple): the coords where the value is added to the array
        """
        if len(coords) != len(self.__shape):
            Exception("len(coords) != len(self.__shape)")
        else:
            pass


class GraphField(FieldContent):
    """
    Class for represent a graph-field.
    """
    def __init__(self):
        super().__init__()


class ItemField:
    """
    A field of an item can be represented in different ways, indexed in representation_list.
    Args:
        field_name (str): the name of the field
        representations_list (list<FieldContent>): the list of the representations.
    """
    def __init__(self, field_name: str, representations_list: list[FieldContent] = None):
        if representations_list is None:
            representations_list = []
        self.__field_name: str = field_name
        self.__representations_list: list = representations_list

    def __eq__(self, other):
        """
        override of the method __eq__ of object.

        Args:
            other (ItemField): another field

        Returns:
            True if the names are the same
        """
        return self.__field_name == other.__field_name

    def append(self, representation: FieldContent):
        """
        Append a new representation to the representation_list

        Args:
            representation (FieldContent): a representation
        """
        self.__representations_list.append(representation)

    def get_name(self):
        return self.__field_name
