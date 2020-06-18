from abc import ABC, abstractmethod


class LODPropertiesRetrieval(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def retrieve(self):
        pass

