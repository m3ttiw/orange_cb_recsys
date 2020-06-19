from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd


class Graph(ABC):
    """
    Abstract class that generalize the concept of a Graph
    Attributes:
        source_frame (pandas.DataFrame): must contains at least 'from', 'to', 'score' columns. The graph will be
            generated from this DataFrame
    """
    def __init__(self, source_frame: pd.DataFrame):
        self.__graph = None
        if self.__check_columns(source_frame):
            self.create_graph()
            for idx, row in source_frame.iterrows():
                self.add_edge(row['from'], row['to'], self.normalize_score(row['score']))
        else:
            raise ValueError('The source frame must contains at least \'from\', \'to\', \'score\' columns')

    @staticmethod
    def __check_columns(df: pd.DataFrame):
        """
        Check if there are at least least 'from', 'to', 'score' columns in the DataFrame
        Args:
            df (pandas.DataFrame): DataFrame to check

        Returns:
            bool: False if there aren't 'from', 'to', 'score' columns, else True
        """
        if 'from' not in df.columns or 'to' not in df.columns or 'score' not in df.columns:
            return False
        return True

    @staticmethod
    def normalize_score(score: float) -> float:
        """
        Convert the score in the range [-1.0, 1.0] in a normalized weight [0.0, 1.0]
        Args:
            score (float): float in the range [-1.0, 1.0]

        Returns:
            float in the range [0.0, 1.0]
        """
        return 1 - score / 2

    @abstractmethod
    def create_graph(self):
        raise NotImplementedError

    @abstractmethod
    def add_node(self, node: object):
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, from_node: object, to_node: object, weight: float, attr: List[object] = None):
        """ adds an edge, if the nodes are not in the graph, adds the nodes"""
        raise NotImplementedError

    @abstractmethod
    def get_edge(self, from_node: object, to_node: object):
        """it can be None if does not exist"""
        raise NotImplementedError

    @abstractmethod
    def get_adj(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_predecessors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_successors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError


class BipartiteGraph(Graph):
    def __init__(self, source_frame: pd.DataFrame):
        super().__init__(source_frame)

    @abstractmethod
    def create_graph(self):
        raise NotImplementedError

    @abstractmethod
    def add_node(self, node: object):
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, from_node: object, to_node: object, weight: float, attr: List[object] = None):
        """ adds an edge, if the nodes are not in the graph, adds the nodes"""
        raise NotImplementedError

    @abstractmethod
    def get_edge(self, from_node: object, to_node: object):
        """it can be None if does not exist"""
        raise NotImplementedError

    @abstractmethod
    def get_adj(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_predecessors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_successors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError


class TripariteGraph(Graph):
    """ rating su più fields -> più archi (import di RatingsProcessor)"""
    def __init__(self, source_frame: pd.DataFrame, contents_dir: str = None):
        super().__init__(source_frame)
        self.__contents_dir = contents_dir

    def set_contents_dir(self, contents_dir: str):
        self.__contents_dir = contents_dir

    def get_contents_dir(self) -> str:
        return self.__contents_dir

    @abstractmethod
    def create_graph(self):
        raise NotImplementedError

    @abstractmethod
    def add_node(self, node: object):
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, from_node: object, to_node: object, weight: float, attr: List[object] = None):
        """ adds an edge, if the nodes are not in the graph, adds the nodes"""
        raise NotImplementedError

    @abstractmethod
    def get_edge(self, from_node: object, to_node: object):
        """it can be None if does not exist"""
        raise NotImplementedError

    @abstractmethod
    def get_adj(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_predecessors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_successors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def extract_properties(self):
        raise NotImplementedError
