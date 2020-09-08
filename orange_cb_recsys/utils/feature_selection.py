import networkx as nx
from abc import ABC, abstractmethod

class FeatureSelection(ABC):
    @abstractmethod
    def perform(self, graph):
        raise NotImplementedError


class NXFSPageRank(FeatureSelection):
    def perform(self, graph: nx.Graph):
        new_graph = nx.Graph(nx.pagerank(graph))
        return new_graph