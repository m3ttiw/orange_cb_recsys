import pandas as pd
import os

from orange_cb_recsys.evaluation.novelty import perform_novelty
from orange_cb_recsys.evaluation.ranking_metrics import *
from orange_cb_recsys.evaluation.prediction_metrics import *
from orange_cb_recsys.evaluation.fairness_metrics import *
from orange_cb_recsys.evaluation.serendipity import perform_serendipity
from orange_cb_recsys.utils.const import *


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
    content_prediction = pd.Series(predictions['to_id'].values)

    if "relevant_threshold" in options.keys():
        relevant_rank = truth[truth['rating'] >= options["relevant_threshold"]]
    else:
        relevant_rank = truth

    if "relevance_split" in options.keys():
        relevance_split = options["relevance_split"]
    else:
        relevance_split = None

    content_truth = pd.Series(relevant_rank['to_id'].values)

    results = {
        "Precision": perform_precision(prediction_labels=content_prediction, truth_labels=content_truth),
        "Recall": perform_recall(prediction_labels=content_prediction, truth_labels=content_truth),
        "MRR": perform_MRR(prediction_labels=content_prediction, truth_labels=content_truth),
        "NDCG": perform_NDCG(predictions=predictions, truth=truth, split=relevance_split),
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


def perform_fairness_metrics(score_frame: pd.DataFrame, truth_frame: pd.DataFrame, user_groups: Dict[str, float],
                             algorithm_name: str, file_output_directory: str = '/datasets/evaluation'
                             ) -> (pd.DataFrame, pd.DataFrame, float):
    """
    Perform all 'fairness' metrics

    Args:
        score_frame (pd.DataFrame): each row contains index(the rank position), label, value predicted
        truth_frame (pd.DataFrame): the real rank each row contains index(the rank position), label, value
        user_groups (Dict[str, float): each key contains the name of the group and each value contains the
            percentage of the specified group. If the groups don't cover the entire user collection,
            the rest of the users are considered in a 'default_diverse' group
        algorithm_name (str): name of the algorithm that run these metrics
        file_output_directory (str): output directory for saving the results

    Returns:
        Tuple<pd.DataFrame, pd.DataFrame, float>: results_by_user, results_by_user_groups, catalogue_coverage
    """
    if DEVELOPING:
        output_path = file_output_directory
    else:
        output_path = os.path.join(home_path, 'evaluation_plottings', file_output_directory)
    logger.info("working in dir: {}".format(output_path))

    pop_items = popular_items(score_frame=truth_frame)
    pop_ratio_user = pop_ratio_by_user(score_frame=score_frame, most_pop_items=pop_items)

    df_gini = perform_gini_index(score_frame=score_frame)

    logger.info("Splitting users in groups")
    user_groups = split_user_in_groups(score_frame=score_frame, groups=user_groups, pop_items=pop_items)
    delta_gap_score = perform_delta_gap(score_frame=score_frame, truth_frame=truth_frame, users_groups=user_groups)
    profile_vs_recs_pop_ratio = perform_pop_ratio_profile_vs_recs(user_groups=user_groups, truth_frame=truth_frame,
                                                                  most_popular_items=pop_items,
                                                                  pop_ratio_by_users=pop_ratio_user,
                                                                  algorithm_name=algorithm_name,
                                                                  out_dir=file_output_directory)
    perform_pop_recs_correlation(truth_frame=truth_frame, score_frame=score_frame, algorithm_name=algorithm_name,
                                 out_dir=output_path)
    perform_recs_long_tail_distr(truth_frame=score_frame, algorithm_name=algorithm_name, out_dir=output_path)
    # results_by_user = pd.merge(df_gini, other_frame, on='from_id')
    results_by_user = df_gini
    results_by_user_group = pd.merge(delta_gap_score, profile_vs_recs_pop_ratio, on='user_group')

    cat_cov = catalog_coverage(score_frame=score_frame, truth_frame=truth_frame)
    return results_by_user, results_by_user_group, cat_cov


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


def perform_serendipity_novelty_metrics(score_frame: pd.DataFrame, truth_frame: pd.DataFrame):
    serendipity = perform_serendipity(score_frame, popular_items(truth_frame))
    novelty = perform_novelty(score_frame, truth_frame)

    return pd.DataFrame.from_records([(serendipity, novelty)], columns=["serendipity", "novelty"])
