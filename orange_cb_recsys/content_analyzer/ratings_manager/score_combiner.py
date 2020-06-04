from typing import List


def avg(score_list: List[float]) -> float:
    return sum(score_list) / len(score_list)


def mode(score_list: List[float]) -> float:
    return max(set(score_list), key=score_list.count)


class ScoreCombiner:
    def __init__(self, function: str):
        self.__function = eval(function)

    def combine(self, score_list: List[float], **kwargs) -> float:
        return self.__function(score_list, **kwargs)
