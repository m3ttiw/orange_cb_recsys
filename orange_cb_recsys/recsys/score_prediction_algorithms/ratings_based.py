import os
import pickle
from sklearn.feature_extraction import DictVectorizer

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import RatingsSPA

import pandas as pd


class CentroidVector(RatingsSPA):
    def __init__(self, item_field: str, field_representation: str):
        super().__init__(item_field, field_representation)

    def predict(self, user: Content, item: Content, ratings: pd.DataFrame, items_directory: str):
        pass


class ClassifierRecommender(RatingsSPA):
    def __init__(self, item_field: str, field_representation: str):
        super().__init__(item_field, field_representation)

    def predict(self, user: Content, item: Content, ratings: pd.DataFrame, items_directory: str):
        items = [filename for filename in os.listdir(items_directory)]

        features_bag_list = []
        rated_item_index_list = []
        for item in items:
            item_filename = items_directory + '/' + item
            with open(item_filename, "rb") as content_file:
                content = pickle.load(content_file)

                features_bag_list.append(content.get_field("Plot").get_representation("1").get_value())
        features_bag_list.append(content.get_field("Plot").get_representation("1").get_value())
        v = DictVectorizer(sparse=False)

        X_tmp = v.fit_transform(features_bag_list)
