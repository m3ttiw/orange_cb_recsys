from abc import ABC, abstractmethod
from scipy import spatial


class Similarity(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def perform(self, v1, v2):
        raise NotImplementedError


class CosineSimilarity(Similarity):
    def __init__(self):
        super().__init__()

    def perform(self, v1, v2):
        return 1 - spatial.distance.cosine(v1, v2)
