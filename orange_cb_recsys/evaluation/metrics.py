from typing import Dict
import pandas as pd
import numpy as np

def find_matching_ratings(s1: pd.Series, s2: pd.Series):
    for i in range (1,len(s1):


def perform_ranking_metrics(predictions: pd.Series, truth: pd.Series) -> Dict[str, object]:
    def perform_precision():
        """
        Calculates the precision of the recommendations provided
        """
        ncorrect = predictions.isin(truth.index).sum()
        return ncorrect/len(predictions)

    def perform_recall():
        """
        Calculates the recall of the recommendations provided
        """
        ncorrect = predictions.isin(truth.index).sum()
        return ncorrect/len(truth)

    def perform_f1():
        """
        Calculates the f1-measure of the recommendations provided
        """
        prec = perform_precision()
        rec = perform_recall()
        return 2 * ((prec * rec)/ (prec + rec))

    def __perform_dcg(scores: pd.Series):
        """
        Calculates the DCG of a given Series of scores
        """
        dcg, i = 0
        for s in scores:
            i += 1
            dcg += s/np.log2(i)
        return dcg

    def perform_ndcg():
        """
        Calculates the NDCG, given by the ratio between the DCG of the recommendations provided and the
        Ideal-DCG, represented by the DCG of the truth base
        """
        dcg = __perform_dcg(predictions)
        ideal_dcg = __perform_dcg(truth)
        return dcg / ideal_dcg

    results = {
        "precision": perform_precision(),
        "recall": perform_recall(),
        "F1": perform_f1(),
        "NDCG": perform_ndcg()
    }
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
