from abc import ABC
from typing import Dict


class RankingAlgorithm(ABC):
    def rank(self, score_dict: Dict[str, float]):
        raise NotImplementedError


class TopNRanking(RankingAlgorithm):
    def __init__(self, n: int):
        self.__n = n

    def rank(self, score_dict: Dict[str, float]):
        return {item_id: score for item_id, score in sorted(score_dict.items(), key=lambda item: item[1])}
