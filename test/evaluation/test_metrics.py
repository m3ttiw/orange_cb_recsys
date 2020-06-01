from unittest import TestCase
from orange_cb_recsys.evaluation.metrics import perform_ranking_metrics
import pandas as pd

class Test(TestCase):
    def test_perform_ranking_metrics(self):

        truth_rank = {
            "item0": 1.0,
            "item1": 1.0,
            "item2": 0.85,
            "item3": 0.8,
            "item4": 0.7,
            "item5": 0.65,
            "item6": 0.4,
            "item7": 0.35,
            "item8": 0.2,
            "item9": 0.2,
        }

        relevant_rank = {item: score for i, (item, score) in enumerate(truth_rank.items()) if score > 0.75}

        predicted_rank = {
            "item2": 0.9,
            "item5": 0.85,
            "item9": 0.75,
            "item0": 0.7,
            "item4": 0.65,
            "item1": 0.5,
            "item8": 0.2,
            "item7": 0.2,
        }

        col = ["item", "score"]

        results = perform_ranking_metrics(
            pd.DataFrame(predicted_rank.items(), columns=col),
            pd.DataFrame(relevant_rank.items(), columns=col)
        )

        results_2 = perform_ranking_metrics(
            pd.DataFrame(predicted_rank.items(), columns=col),
            pd.DataFrame(relevant_rank.items(), columns=col),
            fn=2, ndcg_scikit=True
        )

        results["F2"] = results_2["F2"]

        real_results = {
            "Precision": 0.375,
            "Recall": 0.75,
            "F1": 0.5,
            "F2": 0.417,
            "NDCG": 0,
        }

        tolerance = 0.5
        for metric, real_score in real_results:
            error = abs(results[metric] - real_score)
            self.assertLessEqual(error, tolerance,
                                 "{} tolerance overtaking: error = {}, tolerance = {}".format(metric, error, tolerance))
