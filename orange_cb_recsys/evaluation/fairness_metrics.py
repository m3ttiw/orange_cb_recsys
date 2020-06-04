import pandas as pd
import numpy as np

# fairness_metrics_results = pd.DataFrame(columns=["user", "gini-index", "delta-gaps", "pop_ratio_profile_vs_recs", "pop_recs_correlation", "recs_long_tail_distr"])


def perform_gini_index(score_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Gini index score for each user in the DataFrame
    Args:
        score_frame (pd.DataFrame): frame wich stores ('user_id', 'item_id', 'rating')

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
    for idx, df in score_frame.groupby('user_id'):
        # from the DataFrame exract the 'rating' column as a np.array and create a Gini_index
        # store the score in a dict
        score_dict[df['user_id'].iloc[0]] = gini(df['rating'].to_numpy())
    results = pd.DataFrame({'user': list(score_dict.keys()), 'gini-index': list(score_dict.values())})

    return results


def perform_delta_gaps(score_frame: pd.DataFrame):
    pass


def perform_pop_ratio_profile_vs_recs():
    pass


def perform_pop_recs_correlation():
    pass


def recs_long_tail_distr():
    pass
