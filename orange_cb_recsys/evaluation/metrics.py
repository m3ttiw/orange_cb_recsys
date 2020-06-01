from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def perform_ranking_metrics(predictions: pd.DataFrame,
                            truth: pd.DataFrame,
                            **options) -> Dict[str, object]:
    content_prediction = predictions.iloc[0]
    content_truth = truth.iloc[0]

    def perform_precision():
        """
        Returns the precision of the given ranking (predictions)
        based on the truth ranking
        """
        return len(content_prediction.intersection(content_truth)) / len(content_prediction)

    def perform_recall():
        """
        Returns the recall of the given ranking (predictions)
        based on the truth ranking
        """
        return len(content_prediction.intersection(content_truth)) / len(content_truth)

    def perform_Fn(n: int = 1, precision: float = None, recall: float = None):
        """
        Returns the Fn measure of the given ranking (predictions)
        based on the truth ranking
        """
        p = precision if precision is not None else perform_precision()
        r = recall if recall is not None else perform_recall()
        return (1 + n ** 2) * ((p * r) / (n ** 2 * p + r))

    def perform_DCG(scores: pd.Series):
        """
        Returns the DCG measure of the given ranking (predictions)
        based on the truth ranking
        """
        dcg = 0
        for score, i in enumerate(scores):
            dcg += score / np.log2(i)
        return dcg

    def perform_NDCG():
        """
        Returns the NDCG measure of the given ranking (predictions)
        based on the Ideal DCG of truth ranking
        """
        return perform_DCG(predictions.iloc[1]) / perform_DCG(truth.iloc[1])

    def perform_NDCG_scikit():
        """
        Returns the NDCG measure with scickit learn
        """
        return ndcg_score(truth.iloc[1], predictions.iloc[1], k=len(truth))

    results = {
        "precision": perform_precision(),
        "recall": perform_recall(),
        "NDCG": perform_NDCG_scikit()}

    if options["fn"] is not None and options["fn"] > 1:
        results["F{}".format(options["fn"])] = perform_Fn(n=options["fn"], precision=results["Precision"],
                                                          recall=results["Recall"])
    else:
        results["F1"] = perform_Fn(precision=results["Precision"], recall=results["Recall"])

    return results


def perform_fairness_metrics() -> Dict[str, object]:
    def perform_gini_index():
        pass

    def perform_pop_recs_correlation():
        pass

    results = {}

    results["precision"] = perform_gini_index()
    results["pop_recs_correlation"] = perform_pop_recs_correlation()

    return results


def perform_prediction_metrics(predictions: pd.Series, truth: pd.Series) -> Dict[str, object]:
    def perform_RMSE():
        pass

    def perform_MAE():
        pass

    results = {}

    results["RMSE"] = perform_RMSE()
    results["MAE"] = perform_MAE()

    return results
