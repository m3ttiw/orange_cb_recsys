from abc import ABC
from typing import Dict
import pandas as pd


def perform_ranking_metrics(predictions: pd.DataFrame, truth: pd.DataFrame) -> Dict[str, object]:
    def perform_precision():
        return

    def perform_recall():
        return

    def perform_F1():
        return

    def perform_NDCG():
        return

    results = {}

    results["precision"] = perform_precision()
    results["recall"] = perform_recall()
    results["F1"] = perform_F1()
    results["NDCG"] = perform_NDCG()

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