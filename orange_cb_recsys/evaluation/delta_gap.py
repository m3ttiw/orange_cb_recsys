from typing import List, Dict

import pandas as pd
import numpy as np


def get_avg_pop(items: pd.DataFrame, pop_by_items: Dict[str, object]) -> float:
    """

    Args:
        items:
        pop_by_items:

    Returns:

    """
    total_popularity = 0
    for item in items:
        total_popularity += pop_by_items[item]
    return total_popularity / len(items)


def get_avg_pop_by_users(data: pd.DataFrame,
                         pop_by_items: Dict[str, object],
                         rating_filter: bool = False) -> Dict[str, float]:
    """

    Args:
        rating_filter:
        data:
        pop_by_items:

    Returns:

    """
    users = set(data[['from_id']].values.flatten())
    avg_pop_by_users = {}
    query = 'from_id == @user'
    if rating_filter:
        query = 'from_id == @user and rating > 0.0'
    for user in users:
        user_items = data.query(expr=query)[['to_id']].values.flatten()
        avg_pop_by_users[user] = get_avg_pop(user_items, pop_by_items)

    return avg_pop_by_users


# pop_by_items = Counter(group['item_id'].to_numpy())
# It calculates the Group Average Popularity(GAP)
def calculate_gap(group: pd.DataFrame, pop_by_items: Dict[str, object]) -> float:
    """

    Args:
        group:
        pop_by_items:

    Returns:

    """
    total_pop = 0
    avg_pop_by_users: Dict[str, float] = get_avg_pop_by_users(group, pop_by_items)
    for idx, row in group.iterrows():
        total_pop += avg_pop_by_users[row['from_id']]
    return total_pop / len(group)


def calculate_delta_gap(recs_gap, profile_gap) -> float:
    """

    Args:
        recs_gap:
        profile_gap:

    Returns:

    """
    return (recs_gap - profile_gap) / profile_gap


def pop_ratio_by_user(score_frame: pd.DataFrame) -> pd.DataFrame:
    items = ratings[['item']].values.flatten()

    ratings_counter = Counter(items)

    num_of_items = len(ratings_counter.keys())
    top_n_percentage = 0.2
    top_n_index = round(num_of_items * top_n_percentage)

    # a plot could be produced
    most_common = ratings_counter.most_common(top_n_index)

    # removing counts from most_common
    popular_items = set(map(lambda x: x[0], most_common))

    ################## Splitting users by popularity ####################
    users = set(ratings[['user']].values.flatten())

    popularity_ratio_by_user = {}

    for user in users:
        # filters by the current user and returns all the items he has rated
        rated_items = set(ratings.query('user == @user')[['item']].values.flatten())
        # interesects rated_items with popular_items
        popular_rated_items = rated_items.intersection(popular_items)
        popularity_ratio = len(popular_rated_items) / len(rated_items)

        popularity_ratio_by_user[user] = popularity_ratio


def split_user_in_groups(score_frame: pd.DataFrame,
                         niche_percentage: float = 0.2,
                         bb_focused_percentage: float = 0.8) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Split of dataframe in 3 different dataframes, based on the recomendation poplularity of each user
    Args:
        bb_focused_percentage:
        niche_percentage:
        score_frame:

    Returns:

    """

    def sort_by_popularity_ratio(val):
        return val[1]

    def get_users(users_with_pop_ratio):
        return list(map(lambda x: x[0], users_with_pop_ratio))

    pop_ratio_by_users = pop_ratio_by_user(score_frame)

    pop_ratio_by_users.sort(key=sort_by_popularity_ratio)

    num_of_users = len(pop_ratio_by_users)
    niche_last_index = round(num_of_users * niche_percentage)
    bb_focused_first_index = round(num_of_users * bb_focused_percentage)

    niche_users = get_users(pop_ratio_by_users[:niche_last_index])
    diverse_users = get_users(pop_ratio_by_users[niche_last_index:bb_focused_first_index])
    bb_focused_users = get_users(pop_ratio_by_users[bb_focused_first_index:])

    return niche_users, diverse_users, bb_focused_users

