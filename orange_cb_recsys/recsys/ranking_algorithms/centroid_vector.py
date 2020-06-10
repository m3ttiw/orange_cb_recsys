from typing import Dict

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.recsys.algorithm import RankingAlgorithm
from orange_cb_recsys.recsys.ranking_algorithms.similarities import Similarity
from orange_cb_recsys.content_analyzer.content_representation.content_field import FieldRepresentation
import os
import pandas as pd
import numpy as np

from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.load_content import load_content_instance, get_unrated_items, get_rated_items


class CentroidVector(RankingAlgorithm):
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
            ratings (pd.DataFrame): DataFrame containing the ratings.

        Returns:
            arrays (dict<str, FieldRepresentation>): Dictionary whose keys are the id of the items and the values are
            the embedding arrays corresponding to the requested field
        """
        directory_item_list = [os.path.splitext(filename)[0] for filename in os.listdir(items_directory) if filename != 'search_index']
        arrays = []
        rated_items = get_rated_items(items_directory, ratings)
        for item in rated_items:
            content_id = item.get_content_id()
            if float(ratings[ratings['to_id'] == item.get_content_id()].score) >= self.__threshold:
                if self.get_item_field() not in item.get_field_list():
                    raise ValueError("The field name specified could not be found!")
                else:
                    representation = item.get_field(self.get_item_field()).get_representation(self.get_item_field_representation())
                    if representation is None:
                        raise ValueError("The given representation id wasn't found for the specified field")
                    elif len(representation.get_value().shape) != 1:
                        raise ValueError("The specified representation is not a document embedding, so the centroid"
                                         " can not be calculated")
                    else:
                        arrays.append(representation.get_value())
        return np.array(arrays)

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

    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int, items_directory: str) -> pd.DataFrame:
        """
        For each item:
        1) Takes the embedding arrays
        2) Computes the centroid between the representations. In order to do that, field_representation must
        be a representation that allows the computation of a centroid, otherwise the method will raise an exception;
        3) Determines the similarity between the centroid and the field_representation of the item_field in item.
        Args:
            user_id:
            recs_number (list[Content]): How long the ranking will be
            ratings (pd.DataFrame): Ratings
            items_directory (str): Name of the directory where the items are stored.

        Returns:
             scores (pd.DataFrame): DataFrame whose columns are the ids of the items, and the similarities between the
              items and the centroid
        """
        try:
            logger.info("Retrieving array and putting them in the matrix")
            matrix = self.__get_arrays(items_directory, ratings)

            logger.info("Computing centroid")
            centroid = self.__centroid(matrix)
            columns = ["to_id", "similarity"]
            scores = pd.DataFrame(columns=columns)

            logger.info("Computing similarities")
            unrated_items = get_unrated_items(items_directory, ratings)
            for i, item in enumerate(unrated_items):
                item_id = item.get_content_id()
                item_field_representation = item.get_field(self.get_item_field()).get_representation(
                    self.get_item_field_representation()).get_value()
                similarity = self.__similarity.perform(centroid, item_field_representation)
                scores = pd.concat([scores, pd.DataFrame.from_records([(item_id, similarity)], columns=columns)],
                                   ignore_index=True)

            scores = scores.sort_values(['similarity'], ascending=False).reset_index()
            scores = scores[:recs_number]

            return scores
        except ValueError as v:
            print(str(v))

