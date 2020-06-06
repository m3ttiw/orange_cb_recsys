import os
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import RatingsSPA

import pandas as pd


class CentroidVector(RatingsSPA):
    def __init__(self, item_field: str, field_representation: str):
        super().__init__(item_field, field_representation)

    def predict(self, item: Content, ratings: pd.DataFrame, items_directory: str, item_to_classify):
        """
        1) Goes into items_directory and for each item takes the values corresponding to the field_representation
        of the item_field. For example, if item_field == "Plot" and field_representation == "Document embedding",
        the function will take the "Document embedding" representation of each  "Plot" field for every item;
        2) Computes the weighted centroid between the representations. In order to do that, field_representation must
        be a representation that allows the computation of a centroid, otherwise the method will raise an exception;
        3) Determines the similarity between the centroid and the field_representation of the item_field in item.

        Args:
            item (Content): Item for which the similarity will be computed
            ratings (pd.DataFrame): Ratings
            items_directory (str): Name of the directory where the items are stored.

        Returns:
             ----- similarity (float): The similarity between the item and the other items
        """
        return 5.0


class ClassifierRecommender(RatingsSPA):
    def __init__(self, item_field: str, field_representation: str):
        super().__init__(item_field, field_representation)

    def predict(self, item: Content, ratings: pd.DataFrame, items_directory: str, item_to_classify):
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

        for i in X_tmp:
            if X_tmp[i].get_content_id() in ratings.item_id:
                rated_item_index_list.append(X_tmp[i])

        verified = 0

        for i in rated_item_index_list:
            if rated_item_index_list[i] == ratings[i].item_id:
                verified += 1

        if verified == len(rated_item_index_list):
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(rated_item_index_list, ratings.score)

            return clf.predict(item_to_classify)
