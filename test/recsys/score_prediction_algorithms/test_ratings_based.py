from unittest import TestCase
from orange_cb_recsys.recsys.score_prediction_algorithms.ratings_based import CentroidVector
from orange_cb_recsys.recsys.score_prediction_algorithms.similarities import CosineSimilarity

import pandas as pd
import os
import pickle


class TestCentroidVector(TestCase):
    def test_predict(self):
        sim = CosineSimilarity()
        alg = CentroidVector('Plot', '0', sim)
        ratings = pd.DataFrame.from_records([
            ("A000", "Ace Ventura: When Nature Calls_tt0112281", "sdfgd", 2.0, "54654675"),
            ("A000", "Balto_tt0112453", "sdfgd", 3.0, "54654675"),
            ("A000", "Casino_tt0112641", "sdfgd", 4.0, "54654675"),
            ("A000", "Cutthroat Island_tt0112760", "sdfgd", 2.0, "54654675"),
            ("A000", "Dracula: Dead and Loving It_tt0112896", "sdfgd", 5.0, "54654675"),
            ("A000", "Father of the Bride Part II_tt0113041", "sdfgd", 1.0, "54654675"),
            ("A000", "Four Rooms_tt0113101", "sdfgd", 3.0, "54654675")
        ], columns=["user_id", "item_id", "original_rating", "derived_score", "timestamp"])

        path = "../../../contents/movielens_test1591028175.9454775"
        try:
            file = os.path.join(path, "Sudden Death_tt0114576.bin")
            with open(file, "rb") as content_file:
                item = pickle.load(content_file)
            self.assertEqual(alg.predict(item, ratings, path),
                             0.9279127875850923)
        except FileNotFoundError:
            path = "contents/movielens_test1591028175.9454775"
            file = os.path.join(path, "Sudden Death_tt0114576.bin")
            with open(file, "rb") as content_file:
                item = pickle.load(content_file)
            self.assertEqual(alg.predict(item, ratings, path),
                             0.9279127875850923)
