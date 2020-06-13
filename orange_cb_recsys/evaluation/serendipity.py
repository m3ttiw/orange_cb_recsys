from pathlib import Path
from typing import Set

import pandas as pd

from orange_cb_recsys.evaluation.utils import popular_items
from orange_cb_recsys.utils.const import logger


def perform_serendipity(score_frame: pd.DataFrame, most_popular_items: Set[str] = None, num_of_recs: int = 10) -> float:
    """
    Calculates the serendipity score

    Args:
        score_frame (pd.DataFrame): each row contains index(the rank position), label, value predicted
        most_popular_items (Set<str>): Set contains the most popular label of 'to_id'
        num_of_recs (int): avg number of recommendation per user

    Returns:
        serendipity (float): The serendipity value
    """
    if most_popular_items is None:
        most_popular_items = popular_items(score_frame=score_frame)
    users = set(score_frame[['from_id']].values.flatten())

    pop_ratios_sum = 0
    for user in users:
        recommended_items = score_frame.query('from_id == @user')[['to_id']].values.flatten()
        pop_items_count = 0
        for item in recommended_items:
            if item not in most_popular_items:
                pop_items_count += 1

        pop_ratios_sum += pop_items_count / num_of_recs

    serendipity = pop_ratios_sum / len(users)

    return serendipity
