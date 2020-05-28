from abc import ABC
from typing import Dict
import pandas as pd


class RankingAlgorithm(ABC):
    def rank(self, score_dict: Dict[str, float]):
        raise NotImplementedError


class TopNRanking(RankingAlgorithm):
    def __init__(self, n: int):
        self.__n = n

    def rank(self, score_frame: pd.DataFrame):
        return score_frame.sort_values('rating')[:self.__n]
