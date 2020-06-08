from unittest import TestCase
from orange_cb_recsys.recsys.ranking_algorithms.centroid_vector import CentroidVector, ClassifierRecommender
from orange_cb_recsys.recsys.ranking_algorithms.similarities import CosineSimilarity

import lzma
import pandas as pd
import os
import pickle


class TestCentroidVector(TestCase):
    def test_predict(self):
        sim = CosineSimilarity()
        alg = CentroidVector('Plot', '1', sim)
        ratings = pd.DataFrame.from_records([
            ("A000", "Ace Ventura: When Nature Calls_tt0112281", "sdfgd", 0.99, "54654675"),
            ("A000", "Balto_tt0112453", "sdfgd", 0, "54654675"),
            ("A000", "Casino_tt0112641", "sdfgd", 0.44, "54654675"),
            ("A000", "Cutthroat Island_tt0112760", "sdfgd", -0.68, "54654675"),
            ("A000", "Dracula: Dead and Loving It_tt0112896", "sdfgd", -0.32, "54654675"),
            ("A000", "Father of the Bride Part II_tt0113041", "sdfgd", 0.1, "54654675"),
            ("A000", "Four Rooms_tt0113101", "sdfgd", -0.87, "54654675")
        ], columns=["from_id", "to_id", "original_rating", "score", "timestamp"])

        path = "../../../contents/movielens_test1591453035.7551947"
        items = []
        try:
            file1 = os.path.join(path, "Sudden Death_tt0114576.xz")
            with lzma.open(file1, "rb") as content_file:
                items.append(pickle.load(content_file))

            file2 = os.path.join(path, "Toy Story_tt0114709.xz")
            with lzma.open(file2, "rb") as content_file:
                items.append(pickle.load(content_file))
        except FileNotFoundError:
            path = "contents/movielens_test1591453035.7551947"
            file1 = os.path.join(path, "Sudden Death_tt0114576.xz")
            with lzma.open(file1, "rb") as content_file:
                items.append(pickle.load(content_file))

            file2 = os.path.join(path, "Toy Story_tt0114709.xz")
            with lzma.open(file2, "rb") as content_file:
                items.append(pickle.load(content_file))

        columns = ["item_id", "rating"]
        scores = pd.DataFrame.from_records([("Sudden Death_tt0114576", 0.922455443090693),
                                            ("Toy Story_tt0114709", 0.9319401691396478)],
                                           columns=columns)
        self.assertGreater(alg.predict(items, ratings=ratings, items_directory=path).rating[0], 0)


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

        print(alg.predict([item], ratings=ratings, items_directory=path))
        self.assertGreater(alg.predict([item], ratings=ratings, items_directory=path).rating[0], 0)
