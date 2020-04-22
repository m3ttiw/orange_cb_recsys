from src.offline.content_analyzer.field_content_production_technique import CombiningTechnique


class Centroid(CombiningTechnique):
    def __init__(self, weights: list = None):
        super().__init__()
        self.__weights = weights

    def combine(self):
        pass

# your combining technique
