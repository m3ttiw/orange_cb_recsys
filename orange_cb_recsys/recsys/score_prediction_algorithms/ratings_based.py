from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import RatingsSPA


class CentroidVector(RatingsSPA):
    def __init__(self, item_field: str):
        super().__init__(item_field)

    def predict(self, user: Content, item: Content, ratings_field: str, items_directory: str):
        ratings_values = user.get_field(ratings_field).get_representation(str(0))
        # calcolo centroide
