from typing import List


def avg(score_list: List[float]) -> float:
    score = 0
    for s in score_list:
        score += s
    return score/len(score_list)


class ScoreCombiner:
    def __init__(self, function):
        self.__function = function

    def combine(self, score_list: List[float]) -> float:
        return self.__function(score_list)
