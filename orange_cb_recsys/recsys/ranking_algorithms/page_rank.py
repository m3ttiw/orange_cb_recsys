from typing import List
import networkx as nx
import pandas as pd
import numpy as np
from abc import abstractmethod
from orange_cb_recsys.recsys import RankingAlgorithm
from orange_cb_recsys.recsys.graphs import


class NXPageRank(RankingAlgorithm):
    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None):
        graph =
        #feature selection

        raise NotImplemented


class NXPersonalizedPageRank(RankingAlgorithm):
    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None):
        raise NotImplemented