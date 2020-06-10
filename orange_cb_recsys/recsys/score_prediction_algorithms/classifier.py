from typing import List

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from orange_cb_recsys.content_analyzer.content_representation.content import Content

import pandas as pd

from orange_cb_recsys.recsys.algorithm import ScorePredictionAlgorithm
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.load_content import get_rated_items


class ClassifierRecommender(ScorePredictionAlgorithm):
    """
       Class that implements a logistic regression classifier.
       Args:
           item_field (str): Name of the field that contains the content to use
           field_representation (str): Id of the field_representation content
       """
    def __init__(self, item_field: str, field_representation: str, threshold=-1):
        super().__init__(item_field, field_representation)
        self.__threshold = threshold

    def predict(self, user_id: str, items: List[Content], ratings: pd.DataFrame, items_directory: str):
        """
        1) Goes into items_directory and for each item takes the values corresponding to the field_representation of
        the item_field. For example, if item_field == "Plot" and field_representation == "tf-idf", the function will
        take the "tf-idf" representation of each  "Plot" field for every rated item, the tf-idf representation of rated items
        and items to classify will be parsed to dense arrays;
        2) Define target features, items with rating greater (lower) than treshold will be used as positive(negative) examples;
        3) Creates an object Classifier, uses the method fit and predicts the class of the new items

            Args:
                user_id:
                items (List<Content>): Items for which the similarity will be computed
                ratings (pd.DataFrame): Ratings
                items_directory (str): Name of the directory where the items are stored.

            Returns:
                 The predicted classes, or the predict values.
        """

        features_bag_list = []
        logger.info("Retrieving rated items")
        item_instances = get_rated_items(items_directory, ratings)
        if self.__threshold == -1:
            threshold = pd.to_numeric(ratings["score"], downcast="float").mean()
        else:
            threshold = self.__threshold

        score = []
        for item in item_instances:
            if item is not None:
                features_bag_list.append(item.get_field(self.get_item_field()).get_representation(self.get_item_field_representation()).get_value())
                score.append(1 if float(ratings[ratings['to_id'] == item.get_content_id()].score) >= threshold else 0)

        for item in items:
            if item is not None:
                features_bag_list.append(item.get_field(self.get_item_field()).get_representation(self.get_item_field_representation()).get_value())

        logger.info("Parsing bag of words to dense vectors")
        v = DictVectorizer(sparse=False)

        logger.info("Labeling examples")

        X_tmp = v.fit_transform(features_bag_list)
        X = [X_tmp[i] for i in range(0, len(item_instances))]
        clf = LogisticRegression()
        clf = clf.fit(X, score)

        columns = ["to_id", "rating"]
        score_frame = pd.DataFrame(columns=columns)
        new_items = [X_tmp[i] for i in range(len(X), len(features_bag_list))]

        logger.info("Predicting scores")
        scores = clf.predict_proba(new_items)

        for score, item in zip(scores, items):
            if item is not None:
                score_frame = pd.concat([score_frame, pd.DataFrame.from_records([(item.get_content_id(), score[1] * 2 - 1)], columns=columns)], ignore_index=True)

        return score_frame
