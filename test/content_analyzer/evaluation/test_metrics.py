from unittest import TestCase
import pandas as pd

from orange_cb_recsys.evaluation.metrics import *


class MyTestCase(TestCase):
    def test_perform_precision(self):
        pass

    def test_perform_recall(self):
        pass

    def test_perform_f1(self):
        pass

    def test_perform_dcg(self):
        pass

    def test_perform_ndcg(self):
        pass

    def test_perform_ranking_metrics(self):
        pass

    def test_perform_gini_index(self):
        pass

    def test_perform_pop_recs_correlation(self):
        pass

    def test_perform_fairness_metrics(self):
        pass

    def test_perform_rmse(self):
        predictions = pd.Series([5, 5, 4, 3, 3, 2, 1])
        truth = pd.Series([5, 4, 3, 3, 1, 2, 1])
        self.assertEqual(perform_rmse(predictions, truth), 0.9258200997725514)

        predictions = pd.Series([5, 5, 4, 3, 3, 2, 1])
        truth = pd.Series([5, 4, 3, 3, 1, 2])
        with self.assertRaises(Exception):
            perform_rmse(predictions, truth)

    def test_perform_mae(self):
        predictions = pd.Series([5, 5, 4, 3, 3, 2, 1])
        truth = pd.Series([5, 4, 3, 3, 1, 2, 1])
        self.assertEqual(perform_mae(predictions, truth), 0.5714285714285714)

    def test_perform_prediction_metrics(self):
        predictions = pd.Series([5, 4, 4, 3, 3, 2, 1])
        truth = pd.Series([5, 5, 3, 3, 1, 2, 1])
        result ={
            "RMSE": 0.9258200997725514,
            "MAE": 0.5714285714285714
        }
        self.assertEqual(perform_prediction_metrics(predictions, truth), result )
