from src.offline.content_analyzer.field_content_production_technique import CombiningTechnique

from typing import List


class Centroid(CombiningTechnique):
    """"
    Class that implements the Abstract Class CombiningTechnique.
    This class calculate the centroid given the list of weights.

    Args:
        weights (list): list of weights, used to calculate the centroid.
    """
    def __init__(self, weights: List[float] = None):
        super().__init__()
        self.__weights: List[float] = weights

    def combine(self):
        """"
        Implements the Abstract Method combine in Combining Technique.
        """
        pass

# your combining technique
