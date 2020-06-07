from typing import Dict, List

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import RatingsSPA
from orange_cb_recsys.recsys.score_prediction_algorithms.similarities import Similarity
from orange_cb_recsys.content_analyzer.content_representation.content_field import FieldRepresentation
import os
import pandas as pd
import numpy as np

from orange_cb_recsys.utils.load_content import load_content_instance


class CentroidVector(RatingsSPA):
    """
    Class that implements a centroid-like recommender. It first gets the centroid of the items that the user liked.
    Then computes the similarity between the centroid and the item of which predict the score.
    Args:
        item_field: Name of the field that contains the content to use
        field_representation: Id of the field_representation content of which compute the centroid and then the
        similarity (Similarity): Kind of similarity to use
        threshold (int): Threshold for the ratings. If the rating is greater than the threshold, it will be considered
        as positive
    """
    def __init__(self, item_field: str, field_representation: str, similarity: Similarity, threshold: int = 0):
        super().__init__(item_field, field_representation)
        self.__similarity = similarity
        self.__threshold = threshold

    def __get_arrays(self, items_directory: str, ratings: pd.DataFrame) -> Dict[str, FieldRepresentation]:
        """
        1) Iterates the files into items_directory
        2) For each file (that represents an item), checks if its id is present in rated_items. If false, skips
        to the next file. Then checks on the rating. If is smaller than the threshold, skips. Otherwise goes on
        item_field if exists
        3) Checks if the representation corresponding to field_representation exists
        4) Checks if the field representation is a document embedding (whose shape equals 1)
        5) If the previous checks went well, takes the value corresponding to the representation id and adds it to
           a dictionary

        Example: item_field == "Plot" and field_representation == "1", the function will check if the "01"
        representation of each "Plot" field is a document embedding, and then adds the embeddings to the arrays
        dictionary.

        Args:
            items_directory (str): Name of the directory where the items are stored.
            rated_items (list): List of item ids that the user has rated.

        Returns:
            arrays (dict<str, FieldRepresentation>): Dictionary whose keys are the id of the items and the values are
            the embedding arrays corresponding to the requested field
        """
        directory_item_list = [os.path.splitext(filename)[0] for filename in os.listdir(items_directory) if filename != 'search_index']
        arrays: dict = {}
        for item in directory_item_list:
            content = load_content_instance(items_directory, item)
            content_id = content.get_content_id()
            rating = ratings.loc[ratings['to_id'] == content_id].score
            if not rating.empty and rating.item() > self.__threshold:
                if self.get_item_field() not in content.get_field_list():
                    raise ValueError("The field name specified could not be found!")
                else:
                    representation = content.get_field(self.get_item_field()).get_representation(self.get_field_representation())
                    if representation is None:
                        raise ValueError("The given representation id wasn't found for the specified field")
                    elif len(representation.get_value().shape) != 1:
                        raise ValueError("The specified representation is not a document embedding, so the centroid"
                                         " can not be calculated")
                    else:
                        arrays[content_id] = representation.get_value()
        return arrays

    @staticmethod
    def __build_matrix(arrays: dict) -> np.array:
        """
        Builds a matrix containing the values of the arrays stored in arrays

        Args:
            arrays (dict): Dictionary whose values mus be inserted in the matrix

        Returns:
             np.array: Matrix
        """
        matrix = []
        for item_id in arrays.keys():
            matrix.append(arrays[item_id])
        return np.array(matrix)

    @staticmethod
    def __centroid(matrix) -> np.ndarray:
        """
        Calculates the centroid of a matrix

        Args:
            matrix (np.array): The matrix of which calculate the centroid

        Returns:
            np.ndarray: The array representing the centroid of the matrix
        """
        return np.average(matrix, axis=0)

    def predict(self, items: List[Content], ratings: pd.DataFrame, items_directory: str) -> pd.DataFrame:
        """
        For each item:
        1) Takes the embedding arrays
        2) Computes the weighted centroid between the representations. In order to do that, field_representation must
        be a representation that allows the computation of a centroid, otherwise the method will raise an exception;
        3) Determines the similarity between the centroid and the field_representation of the item_field in item.
        Args:
            items (list[Content]): Items for which the similarity will be computed
            ratings (pd.DataFrame): Ratings
            items_directory (str): Name of the directory where the items are stored.

        Returns:
             scores (pd.DataFrame): DataFrame whose columns are the ids of the items, and the similarities between the
              items and the centroid
        """
        try:
            arrays = self.__get_arrays(items_directory, ratings)
            matrix = self.__build_matrix(arrays)
            centroid = self.__centroid(matrix)
            columns = ["item_id", "rating"]
            scores = pd.DataFrame(columns=columns)
            for item in items:
                item_id = item.get_content_id()
                item_field_representation = item.get_field(self.get_item_field()).get_representation(
                    self.get_field_representation()).get_value()
                similarity = self.__similarity.perform(centroid, item_field_representation)
                score = similarity * 2 - 1
                scores = pd.concat([scores, pd.DataFrame.from_records([(item_id, score)], columns=columns)],
                                   ignore_index=True)

            return scores
        except ValueError as v:
            print(str(v))


class ClassifierRecommender(RatingsSPA):
    """
       Class that implements a decisiontreeclassifier.
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
        take the "tf-idf" representation of each  "Plot" field for every item;
        2) Takes a list of ratings that are in the dataframe (rated_item_index_list) and does a trasformation on that
        list with the dictvectorizer;
        3) Creates an object DecisionTreeClassifier, uses the method fit and predicts the class of the item
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
            features_bag_list.append(item.get_field(self.get_item_field()).get_representation(self.get_field_representation()).get_value())

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
