from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def perform_ranking_metrics(predictions: pd.DataFrame,
                            truth: pd.DataFrame,
                            **options) -> Dict[str, float]:
    content_prediction = pd.Series(predictions['item'].values)
    if "relevant_threshold" in options.keys():
        relevant_rank = truth[truth['rating'] >= options["relevant_threshold"]]
    else:
        relevant_rank = truth

    content_truth = pd.Series(relevant_rank['item'].values)

    def perform_precision():
        """
        Returns the precision of the given ranking (predictions)
        based on the truth ranking
        """
        return content_prediction.isin(content_truth).sum() / len(content_prediction)

    def perform_recall():
        """
        Returns the recall of the given ranking (predictions)
        based on the truth ranking
        """
        return content_prediction.isin(content_truth).sum() / len(content_truth)

    def perform_Fn(n: int = 1, precision: float = None, recall: float = None):
        """
        Returns the Fn measure of the given ranking (predictions)
        based on the truth ranking
        """
        p = precision if precision is not None else perform_precision()
        r = recall if recall is not None else perform_recall()
        return (1 + (n ** 2)) * ((p * r) / ((n ** 2) * p + r))

    def perform_DCG(gains: pd.Series):
        """
        Returns the DCG array of a gain vector
        """
        dcg = []
        for i, gain in enumerate(gains):
            if i == 0:
                dcg.append(gain)
            else:
                dcg.append((gain / np.log2(i+1)) + dcg[i - 1])
        return dcg

    def perform_NDCG():
        """
        Returns the NDCG measure using Truth rank as ideal DCG
        """

        idcg = perform_DCG(pd.Series(truth['rating'].values))

        col = ["item", "rating"]
        new_predicted = pd.DataFrame(columns=col)
        for index, predicted_row in predictions.iterrows():
            predicted_item = predicted_row['item']
            truth_row = truth.loc[truth['item'] == predicted_item]
            truth_score = truth_row['rating'].values[0]
            new_predicted = new_predicted.append({'item': predicted_item, 'rating': truth_score}, ignore_index=True)

        dcg = perform_DCG(pd.Series(new_predicted['rating'].values))
        ndcg = []
        for i, ideal in enumerate(idcg):
            try:
                ndcg.append(dcg[i] / ideal)
            except IndexError:
                break
        return ndcg

    results = {
        "Precision": perform_precision(),
        "Recall": perform_recall(),
        "NDCG": perform_NDCG(),
    }

    if "fn" in options.keys() and options["fn"] > 1:
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
