import pandas as pd
import numpy as np
from orange_cb_recsys.evaluation.delta_gap import *


# fairness_metrics_results = pd.DataFrame(columns=["user", "gini-index", "delta-gaps", "pop_ratio_profile_vs_recs", "pop_recs_correlation", "recs_long_tail_distr"])


def perform_gini_index(score_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Gini index score for each user in the DataFrame
    Args:
        score_frame (pd.DataFrame): frame wich stores ('from_id', 'to_id', 'rating')

    Returns:
        results (pd.DataFrame): each row contains ('user_id', 'gini_index')
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
    results = pd.DataFrame({'user': list(score_dict.keys()), 'gini-index': list(score_dict.values())})

    return results


def perform_delta_gap(score_frame: pd.DataFrame, **options) -> Dict[str, float]:
    """
    Compute the Delta - GAP (Group Average Popularity) metric
    Args:
        score_frame (pd.DataFrame): frame wich stores ('from_id', 'to_id', 'rating')
    Returns:
        results (pd.DataFrame): each row contains ('user_id', 'delta-gap')
    """

    # results = pd.DataFrame(columns=['user', 'delta-gap-category']) # scrivi per ogni utente a che categoria appartiene
    items = score_frame[['to_id']].values.flatten()
    pop_by_items = Counter(items)
    avg_pop_by_users_profiles = get_avg_pop_by_users(score_frame, pop_by_items)  # truth?
    recs_avg_pop_by_users = get_avg_pop_by_users(score_frame, pop_by_items, rating_filter=True)

    # recommended_users = set(recs[['from_id']].values.flatten())
    recommended_users = set(score_frame.query(expr='rating > 0.0')[['from_id']].values.flatten())

    niche_users, diverse_users, bb_focused_users = split_user_in_groups(score_frame=score_frame, **options)

    niche_delta_gap = calculate_delta_gap(recs_gap=calculate_gap(group=niche_users.intersection(recommended_users),
                                                                 pop_by_items=recs_avg_pop_by_users),
                                          profile_gap=calculate_gap(group=niche_users,
                                                                    pop_by_items=avg_pop_by_users_profiles))
    diverse_delta_gap = calculate_delta_gap(recs_gap=calculate_gap(group=diverse_users.intersection(recommended_users),
                                                                   pop_by_items=recs_avg_pop_by_users),
                                            profile_gap=calculate_gap(group=diverse_users,
                                                                      pop_by_items=avg_pop_by_users_profiles))
    bb_delta_gap = calculate_delta_gap(recs_gap=calculate_gap(group=bb_focused_users.intersection(recommended_users),
                                                              pop_by_items=recs_avg_pop_by_users),
                                       profile_gap=calculate_gap(group=bb_focused_users,
                                                                 pop_by_items=avg_pop_by_users_profiles))

    return {"niche": niche_delta_gap, "diverse": diverse_delta_gap, "bb_focused": bb_delta_gap}

    # for group in split_user_in_groups(score_frame=score_frame, **options):
    #    profile_gap = calculate_gap(group=niche_users, pop_by_items=avg_pop_by_users_profiles)
    #    recs_gap = calculate_gap(group=group_recommended_users, pop_by_items=recs_avg_pop_by_users)
    #    delta_gap = calculate_delta_gap(profile_gap=profile_gap, recs_gap=recs_gap)


def perform_pop_ratio_profile_vs_recs():
    pass


def perform_pop_recs_correlation():
    pass


def recs_long_tail_distr():
    pass


def catalog_coverage():
    pass
