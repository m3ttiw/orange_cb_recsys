from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import RatingsSPA

import pandas as pd


class CentroidVector(RatingsSPA):
    def __init__(self, item_field: str):
        super().__init__(item_field)

    def predict(self, user: Content, item: Content, ratings: pd.DataFrame, items_directory: str):
        pass