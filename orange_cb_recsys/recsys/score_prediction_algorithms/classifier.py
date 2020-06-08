import os
from typing import List

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from orange_cb_recsys.content_analyzer.content_representation.content import Content

import pandas as pd

from orange_cb_recsys.recsys.algorithm import ScorePredictionAlgorithm
from orange_cb_recsys.utils.load_content import load_content_instance


class ClassifierRecommender(ScorePredictionAlgorithm):
    """
       Class that implements a logistic regression classifier.
       Args:
           item_field (str): Name of the field that contains the content to use
           field_representation (str): Id of the field_representation content
       """
    def __init__(self, item_field: str, field_representation: str):
        super().__init__(item_field, field_representation)

    def predict(self, items: List[Content], ratings: pd.DataFrame, items_directory: str):
        """
        1) Goes into items_directory and for each item takes the values corresponding to the field_representation of
        the item_field. For example, if item_field == "Plot" and field_representation == "tf-idf", the function will
        take the "tf-idf" representation of each  "Plot" field for every item, the tf-idf representation will
        be parsed to dense arrays;
        2) Takes a list of ratings that are in the dataframe (rated_item_index_list) and retrieves the corresponding
        dense vectors, items with rating greater (lower) than treshold
        will be used as positive(negative) examples;
        3) Creates an object Classifier, uses the method fit and predicts the class of the item

                Args:
                    items (List<Content>): Items for which the similarity will be computed
                    ratings (pd.DataFrame): Ratings
                    items_directory (str): Name of the directory where the items are stored.

                Returns:
                     The predicted classes, or the predict values.
                """

        features_bag_list = []
        rated_item_index_list = []
        to_classify_item_dict = {item.get_content_id(): None for item in items}

        directory_filename_list = [os.path.splitext(filename)[0] for filename in os.listdir(items_directory) if filename != 'search_index']
        for i, item_filename in enumerate(directory_filename_list):
            item = load_content_instance(items_directory, item_filename)
            features_bag_list.append(item.get_field(self.get_item_field()).get_representation(self.get_item_field_representation()).get_value())

            if ratings['to_id'].str.contains(item.get_content_id()).any():
                rated_item_index_list.append(i)

            if item.get_content_id() in to_classify_item_dict.keys():
                to_classify_item_dict[item.get_content_id()] = i

        v = DictVectorizer(sparse=False)

        score = [1 if rating > 0 else 0 for rating in list(ratings.score)]

        X_tmp = v.fit_transform(features_bag_list)
        X = []
        for item_id in rated_item_index_list:
            X.append(X_tmp[item_id])

        clf = LogisticRegression()
        clf = clf.fit(X, score)

        columns = ["item_id", "rating"]
        score_frame = pd.DataFrame(columns=columns)
        new_items = [X_tmp[to_classify_item_dict[item.get_content_id()]] for item in items]
        scores = clf.predict_proba(new_items)

        for score, item in zip(scores, items):
            score_frame = pd.concat([score_frame, pd.DataFrame.from_records([(item.get_content_id(), score[1])], columns=columns)], ignore_index=True)

        return score_frame
