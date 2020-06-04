from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import RatingsSPA
import os
import pandas as pd
import pickle

class CentroidVector(RatingsSPA):
    def __init__(self, item_field: str, field_representation: str):
        super().__init__(item_field, field_representation)

    def get_arrays(self, items_directory: str):
        os.chdir(items_directory)
        for file in os.listdir():
            with open(file, "rb") as content_file:
                content: Content = pickle.load(content_file)


    def predict(self, item: Content, ratings: pd.DataFrame, items_directory: str):
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
        return
