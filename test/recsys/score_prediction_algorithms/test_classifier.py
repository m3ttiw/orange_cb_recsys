from unittest import TestCase

import lzma
import pandas as pd
import os
import pickle

from orange_cb_recsys.recsys.score_prediction_algorithms.classifier import ClassifierRecommender


class TestClassifierRecommender(TestCase):
    def test_predict(self):

        alg = ClassifierRecommender("Plot", "2")
        ratings = pd.DataFrame.from_records([
            ("A000", "Sudden Death_tt0114576", 0.5, "54654675"),
            ("A000", "Balto_tt0112453", -0.5, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        try:
            path = "../../../contents/movielens_test1591453035.7551947"
            file = os.path.join(path, "Sudden Death_tt0114576.xz")
            with lzma.open(file, "r") as content_file:
                pass
        except FileNotFoundError:
            path = "contents/movielens_test1591453035.7551947"
            file = os.path.join(path, "Sudden Death_tt0114576.xz")

        with lzma.open(file, "r") as content_file:
            item = pickle.load(content_file)

        self.assertGreater(alg.predict('A000', [item], ratings=ratings, items_directory=path).rating[0], 0)
