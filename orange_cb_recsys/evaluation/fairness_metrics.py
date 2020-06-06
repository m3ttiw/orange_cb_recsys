from collections import Counter

import pandas as pd
import numpy as np
from orange_cb_recsys.evaluation.delta_gap import *
from orange_cb_recsys.evaluation.utils import split_user_in_groups


# fairness_metrics_results = pd.DataFrame(columns=["user", "gini-index", "delta-gaps", "pop_ratio_profile_vs_recs", "pop_recs_correlation", "recs_long_tail_distr"])


def perform_gini_index(score_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Gini index score for each user in the DataFrame
    Args:
        score_frame (pd.DataFrame): frame wich stores ('from_id', 'to_id', 'rating')

    Returns:
        results (pd.DataFrame): each row contains ('from_id', 'gini_index')
    """

    def gini(array: np.array):
        """Calculate the Gini coefficient of a numpy array."""

        array = array.flatten()  # all values are treated equally, arrays must be 1d
        if np.amin(array) < 0:
            array += 1
        array += 0.0000001  # values cannot be 0
        array = np.sort(array)  # values must be sorted
        index = np.arange(1, array.shape[0] + 1)  # index per array element
        n = array.shape[0]  # number of array elements
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))  # Gini coefficient

    score_dict = {}
    # for each user extract a pd.DataFrame df
    for idx, df in score_frame.groupby('from_id'):
        # from the DataFrame exract the 'rating' column as a np.array and create a Gini_index
        # store the score in a dict
        score_dict[df['from_id'].iloc[0]] = gini(df['rating'].to_numpy())
    results = pd.DataFrame({'from': list(score_dict.keys()), 'gini-index': list(score_dict.values())})

    return results


def perform_delta_gap(score_frame: pd.DataFrame, truth_frame: pd.DataFrame,
                      users_groups: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Compute the Delta - GAP (Group Average Popularity) metric
    Args:
        truth_frame (pd.DataFrame): frame wich stores ('from_id', 'to_id', 'rating') of user profiles
        score_frame (pd.DataFrame): frame wich stores ('from_id', 'to_id', 'rating') recommended
        users_groups (Dict[str, Set[str]]): each key contains the name of the group and each value the set of 'from_id'
    Returns:
        results (pd.DataFrame): each row contains ('from_id', 'delta-gap')
    """

    # results = pd.DataFrame(columns=['from', 'delta-gap-category']) # scrivi per ogni utente a che categoria appartiene
    items = score_frame[['to_id']].values.flatten()
    pop_by_items = Counter(items)
    avg_pop_by_users_profiles = get_avg_pop_by_users(truth_frame, pop_by_items)  # truth?
    recs_avg_pop_by_users = get_avg_pop_by_users(score_frame, pop_by_items)

    recommended_users = set(truth_frame[['from_id']].values.flatten())
    # recommended_users = set(score_frame.query(expr='rating > 0.0')[['from_id']].values.flatten())

    #niche_users, diverse_users, bb_focused_users = split_user_in_groups(score_frame=score_frame, **options)
    score_dict = {}
    for group_name in users_groups:
        # print("{}: {}".format(group_name, group))
        recs_gap = calculate_gap(group=users_groups[group_name].intersection(recommended_users), avg_pop_by_users=recs_avg_pop_by_users)
        profile_gap = calculate_gap(group=users_groups[group_name], avg_pop_by_users=avg_pop_by_users_profiles)
        group_delta_gap = calculate_delta_gap(recs_gap=recs_gap, profile_gap=profile_gap)
        score_dict[group_name] = group_delta_gap
    return score_dict


def perform_pop_ratio_profile_vs_recs():
    # fetching pop_ratio_by_users, niche, diverse and bb-focused users
    pop_ratio_by_users = pd.read_csv('../datasets/pop-ratio-by-user.csv')
    niche = pd.read_csv('../datasets/niche.csv').values.flatten()
    diverse = pd.read_csv('../datasets/diverse.csv').values.flatten()
    bb_focused = pd.read_csv('../datasets/bb-focused.csv').values.flatten()

    # fetching set of most popular items
    most_popular_items = set(pd.read_csv('../datasets/most-popular-items.csv').values.flatten())

    # calculating ratios of popular items in niche, diverse and bb_focused profiles
    niche_profile_pop_ratios = get_profile_pop_ratios(niche, pop_ratio_by_users)
    diverse_profile_pop_ratios = get_profile_pop_ratios(diverse, pop_ratio_by_users)
    bb_focused_profile_pop_ratios = get_profile_pop_ratios(bb_focused, pop_ratio_by_users)

    # calculating ratios of popular items in niche, diverse and bb_focused recommendations
    recs = recs[['user', 'item']]
    niche_recs_pop_ratios = get_recs_pop_ratios(niche, recs, most_popular_items)
    diverse_recs_pop_ratios = get_recs_pop_ratios(diverse, recs, most_popular_items)
    bb_focused_recs_pop_ratios = get_recs_pop_ratios(bb_focused, recs, most_popular_items)


def perform_pop_recs_correlation():
    pass


def recs_long_tail_distr():
    pass


def catalog_coverage():
    pass
