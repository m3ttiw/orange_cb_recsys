from typing import List, Tuple

from orange_cb_recsys.recsys.graphs import BipartiteGraph
import pandas as pd
import networkx as nx


class NXBipartiteGraph(BipartiteGraph):
    def __init__(self, source_frame: pd.DataFrame):
        super().__init__(source_frame)

    def create_graph(self):
        self.__graph = nx.Graph()

    def add_node(self, node: object):
        self.__graph.add_node(node)

    def add_edge(self, from_node: object, to_node: object, weight: float, attr: List[object] = None):
        self.__graph.add_edge(from_node, to_node, weight=weight)

    def get_adj(self, node: object) -> List[Tuple[object, object, float]]:
        return self.__graph.adj

    def get_predecessors(self, node: object) -> List[Tuple[object, object, float]]:
        pass

    def get_successors(self, node: object) -> List[Tuple[object, object, float]]:
        pass
