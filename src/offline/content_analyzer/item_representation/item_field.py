from abc import ABC
import numpy as np


class FieldContent(ABC):
    def __init__(self):
        pass


class FeaturesBagField(FieldContent):
    def __init__(self, features: dict = None):
        super().__init__()
        if features is None:
            features = {}
        self.__features: dict = features

    def add_feature(self, feature_key: str, feature_value):
        pass


class EmbeddingField(FieldContent):
    def __init__(self, shape: tuple):
        super().__init__()
        self.__shape: tuple = shape
        self.__embedding_array: np.ndarray = np.ndarray(shape=self.__shape)

    def add_value(self, value: float, coords: tuple):
        if len(coords) != len(self.__shape):
            Exception("len(coords) != len(self.__shape)")
        else:
            pass


class GraphField(FeaturesBagField):
    def __init__(self):
        super().__init__()


class ItemField:
    def __init__(self, field_name: str, representations_list: list = None):
        if representations_list is None:
            representations_list = []
        self.__field_name: str = field_name
        self.__representations_list: list = representations_list

    def __eq__(self, other):
        return self.__field_name == other.__field_name

    def append(self, representation: FieldContent):
        self.__representations_list.append(representation)

    def name(self):
        return self.__field_name
