from typing import Set, Dict
import pandas as pd
from collections import Counter


def pop_ratio_by_user(score_frame: pd.DataFrame) -> pd.DataFrame:
    items = score_frame[['to_id']].values.flatten()

    ratings_counter = Counter(items)

    num_of_items = len(ratings_counter.keys())
    top_n_percentage = 0.2
    top_n_index = round(num_of_items * top_n_percentage)

    # a plot could be produced
    most_common = ratings_counter.most_common(top_n_index)

    # removing counts from most_common
    popular_items = set(map(lambda x: x[0], most_common))

    ################## Splitting users by popularity ####################
    users = set(score_frame[['from_id']].values.flatten())

    popularity_ratio_by_user = {}

    for user in users:
        # filters by the current user and returns all the items he has rated
        rated_items = set(score_frame.query('from_id == @user')[['to_id']].values.flatten())
        # interesects rated_items with popular_items
        popular_rated_items = rated_items.intersection(popular_items)
        popularity_ratio = len(popular_rated_items) / len(rated_items)

        popularity_ratio_by_user[user] = popularity_ratio
    return pd.DataFrame.from_dict({'from_id': list(popularity_ratio_by_user.keys()),
                                   'popularity': list(popularity_ratio_by_user.values())})


def split_user_in_groups(score_frame: pd.DataFrame, groups: Dict[str, float]) -> Dict[str, Set[str]]:
    """
    Split of DataFrames in 3 different Sets, based on the recommendation popularity of each user
    Args:
        score_frame (pd.DataFrame): DataFrame with columns = ['from_id', 'to_id', 'rating']
        groups (Dict[str, float]): each key contains the name of the group and each value contains the percentage
                                   of the specified group. If the groups don't cover the entire user collection,
                                   the rest of the users are considered in a 'default_diverse' group

    Returns:
        groups_dict
    """

    pop_ratio_by_users = pop_ratio_by_user(score_frame)
    pop_ratio_by_users.sort_values(['popularity'], inplace=True, ascending=False)
    num_of_users = len(pop_ratio_by_users)
    groups_dict: Dict[str, Set[str]] = {}
    first_index = 0
    last_index = first_index
    percentage = 0.0
    for group_name in groups:
        percentage += groups[group_name]
        group_index = round(num_of_users * percentage)
        groups_dict[group_name] = set(pop_ratio_by_users['from_id'][last_index:group_index])
        last_index = group_index
    if percentage < 1.0:
        group_index = round(num_of_users)
        groups_dict['default_diverse'] = set(pop_ratio_by_users['from_id'][last_index:group_index])
    return groups_dict
