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
        self.__trigraph: NXTripartiteGraph = None

    @property
    def trigraph(self):
        return self.__trigraph

    def set_trigraph(self, graph):
        self.__trigraph = graph

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
        new_rank = {k: rank[k] for k in rank.keys()}
        for k in rank.keys():
            if remove_from_nodes and self.__trigraph.is_from_node(k):
                new_rank.pop(k)
            if remove_profile and self.__trigraph.is_to_node(k) and k in extracted_profile.keys():
                new_rank.pop(k)
            if remove_properties and not self.__trigraph.is_to_node(k) and not self.__trigraph.is_from_node(k):
                new_rank.pop(k)
        return new_rank

    def extract_profile(self, user_id: str) -> Dict:
        adj = self.__trigraph.get_adj(user_id)
        profile = {}
        #logger.info('unpack %s', str(adj))
        for a in adj:
            #logger.info('unpack %s', str(a))
            edge_data = self.__trigraph.get_edge_data(user_id, a)
            profile[a] = edge_data['weight']
            logger.info('unpack %s, %s', str(a), str(profile[a]))
        return profile #{t: w for (f, t, w) in adj}


class NXPageRank(PageRankAlg):

    def __init__(self):
        super().__init__()

    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int,
                items_directory: str,                       # not used
                candidate_item_id_list: List = None):       # not used
        self.set_trigraph(NXTripartiteGraph(ratings))
        # feature selection (TO DO)
        # run the pageRank
        if self.personalized:
            profile = self.extract_profile(user_id)
            scores = nx.pagerank(self.trigraph.graph, personalization=profile)
        else:
            scores = nx.pagerank(self.trigraph.graph)
        # clean the results removing user nodes, selected user profile and eventually properties
        scores = self.clean_rank(scores, user_id)
        ks = scores.keys()
        ks = ks[:recs_number]
        new_scores = {k: scores[k] for k in scores.keys() if k in ks}

        return new_scores
