from typing import List, Dict
import networkx as nx
import pandas as pd
import numpy as np
from abc import abstractmethod
from orange_cb_recsys.recsys import RankingAlgorithm, Graph


class PageRank(RankingAlgorithm):
    def __init__(self, personalized: bool = True):
        super().__init__('', '')
        self.__personalized = personalized
        self.__graph: Graph = None

    @abstractmethod
    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None):
        raise NotImplemented

    def __clean_rank(self, rank: Dict, user_id: str,
                     remove_from_nodes: bool = True,
                     remove_profile: bool = True,
                     remove_properties: bool = True) -> Dict:
        extracted_profile = self.__extract_profile(user_id)
        for k in rank.keys():
            if remove_from_nodes and self.__graph.is_from_node(k):
                rank.pop(k)
            if remove_profile and self.__graph.is_to_node(k) and k in extracted_profile.keys():
                rank.pop(k)
            if remove_properties and not self.__graph.is_to_node(k) and not self.__graph.is_from_node(k):
                rank.pop(k)
        return rank

    def __extract_profile(self, user_id: str) -> Dict:
        adj = self.__graph.get_adj(user_id)
        return {t: w for f, t, w in adj}


class NXPageRank(PageRank):

    def __init__(self):
        super().__init__()

    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int,
                items_directory: str,                       # not used
                candidate_item_id_list: List = None):       # not used
        # create the graph
        if self.__graph is None:
            self.__graph = Graph(ratings)
        # feature selection (TO DO)
        # run the pageRank
        if self.__personalized:
            profile = self.__extract_profile(user_id)
            scores = nx.pagerank(self.__graph, personalization=profile)
        else:
            scores = nx.pagerank(self.__graph)
        # clean the results removing user nodes, selected user profile and eventually properties
        scores = self.__clean_rank(scores, user_id)
        scores = scores[:recs_number]
        return scores
