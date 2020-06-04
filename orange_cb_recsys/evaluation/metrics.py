from typing import Dict
import pandas as pd
from orange_cb_recsys.evaluation.ranking_metrics import *
from orange_cb_recsys.evaluation.prediction_metrics import *
from orange_cb_recsys.evaluation.fairness_metrics import *


def perform_ranking_metrics(predictions: pd.DataFrame,
                            truth: pd.DataFrame,
                            **options) -> Dict[str, float]:
    """
    Perform the computation of all ranking metrics

    Args:
        predictions (pd.DataFrame): each row contains index(the rank position), label, value predicted
        truth (pd.DataFrame): the real rank each row contains index(the rank position), label, value
        **options : you can specify some option parameters like:
         - fn (int): the n of the Fn metric, default = 1

    Returns:
        results (Dict[str, object]): results of the computations of all ranking metrics
    """
    content_prediction = pd.Series(predictions['item_id'].values)
    if "relevant_threshold" in options.keys():
        relevant_rank = truth[truth['rating'] >= options["relevant_threshold"]]
    else:
        relevant_rank = truth

    content_truth = pd.Series(relevant_rank['item_id'].values)

    results = {
        "Precision": perform_precision(prediction_labels=content_prediction, truth_labels=content_truth),
        "Recall": perform_recall(prediction_labels=content_prediction, truth_labels=content_truth),
        "MRR": perform_MRR(prediction_labels=content_prediction, truth_labels=content_truth),
        "NDCG": perform_NDCG(predictions=predictions, truth=truth),
        "pearson": perform_correlation(prediction_labels=content_prediction, truth_labels=content_truth),
        "kendall": perform_correlation(prediction_labels=content_prediction, truth_labels=content_truth,
                                       method='kendall'),
        "spearman": perform_correlation(prediction_labels=content_prediction, truth_labels=content_truth,
                                        method='spearman'),
    }

    if "fn" in options.keys() and options["fn"] > 1:
        results["F{}".format(options["fn"])] = perform_Fn(n=options["fn"], precision=results["Precision"],
                                                          recall=results["Recall"])
    else:
        results["F1"] = perform_Fn(precision=results["Precision"], recall=results["Recall"])

    return results


def perform_fairness_metrics(score_frame: pd.DataFrame) -> pd.DataFrame:
    # results = {
    #    "gini_index": perform_gini_index(score_frame=score_frame),
    #    "pop_recs_correlation": perform_pop_recs_correlation()
    # }
    columns = ["user", "gini-index", "delta-gaps", "pop_ratio_profile_vs_recs","pop_recs_correlation",
               "recs_long_tail_distr"]
    # results = pd.concat([results, perform_gini_index(score_frame=score_frame)], ignore_index=True, axis=1)
    df_gini = perform_gini_index(score_frame=score_frame)

    results = pd.concat([df_gini, ], axis=1)

    print(results)
    return results


def perform_prediction_metrics(predictions: pd.Series, truth: pd.Series) -> Dict[str, object]:
    """
    Performs the metrics for evaluating the rating prediction phase and returns their values

    Args:
        predictions (pd.Series): Series containing the predicted ratings
        truth (pd.Series): Series containing the truth rating values

    Returns:
        results (Dict[str, object]): Python dictionary where the keys are the names of the metrics and the
        values are the corresponding values
    """
    results = {
        "RMSE": perform_rmse(predictions, truth),
        "MAE": perform_mae(predictions, truth)
    }
    return results
