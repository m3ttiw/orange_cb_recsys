import lzma
import os
import pandas as pd
from unittest import TestCase

from orange_cb_recsys.recsys import NXPageRank


class TestNXPageRank(TestCase):
    def test_predict(self):
        alg = NXPageRank()
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 0.5, "54654675"),
            ("A000", "tt0112453", -0.5, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        try:
            path = "../../../contents/movielens_test1591885241.5520566"
            file = os.path.join(path, "tt0114576.xz")
            with lzma.open(file, "r") as content_file:
                pass
        except FileNotFoundError:
            path = "contents/movielens_test1591885241.5520566"

        alg.predict('A000', ratings, 1, path, ['tt0114576'])
