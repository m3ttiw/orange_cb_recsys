from typing import List, Dict
import networkx as nx
import pandas as pd
import numpy as np
from abc import abstractmethod

from orange_cb_recsys.recsys.algorithm import RankingAlgorithm
from orange_cb_recsys.recsys.graphs import Graph
from orange_cb_recsys.recsys.graphs.tripartite_graphs import NXTripartiteGraph
from orange_cb_recsys.utils.const import logger


class PageRankAlg(RankingAlgorithm):
    def __init__(self, personalized: bool = True):
        super().__init__('', '')
        self.__personalized = personalized
        self.__graph: NXTripartiteGraph = None

    @property
    def graph(self):
        return self.__graph

    def set_graph(self, graph):
        self.__graph = graph

    @property
    def personalized(self):
        return self.__personalized

    def set_personalized(self, personalized):
        self.__personalized = personalized

    @abstractmethod
    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None):
        raise NotImplemented

    def clean_rank(self, rank: Dict, user_id: str,
                        remove_from_nodes: bool = True,
                        remove_profile: bool = True,
                        remove_properties: bool = True) -> Dict:
        extracted_profile = self.extract_profile(user_id)
        for k in rank.keys():
            if remove_from_nodes and self.__graph.is_from_node(k):
                rank.pop(k)
            if remove_profile and self.__graph.is_to_node(k) and k in extracted_profile.keys():
                rank.pop(k)
            if remove_properties and not self.__graph.is_to_node(k) and not self.__graph.is_from_node(k):
                rank.pop(k)
        return rank

    def extract_profile(self, user_id: str) -> Dict:
        adj = self.__graph.get_adj(user_id)
        logger.info('unpack %s', str(adj))
        return {t: w for f, t, w in adj}


class NXPageRank(PageRankAlg):

    def __init__(self):
        super().__init__()

    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int,
                items_directory: str,                       # not used
                candidate_item_id_list: List = None):       # not used
        # create the graph
        if self.graph is None:
            self.set_graph(NXTripartiteGraph(ratings))
        # feature selection (TO DO)
        # run the pageRank
        if self.personalized:
            profile = self.extract_profile(user_id)
            scores = nx.pagerank(self.__graph, personalization=profile)
        else:
            scores = nx.pagerank(self.__graph)
        # clean the results removing user nodes, selected user profile and eventually properties
        scores = self.clean_rank(scores, user_id)
        scores = scores[:recs_number]
        return scores
