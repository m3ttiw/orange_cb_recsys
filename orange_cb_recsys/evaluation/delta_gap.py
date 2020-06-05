from collections import Counter
from typing import Dict, Set

import pandas as pd


def get_avg_pop(items: pd.Series, pop_by_items: Dict[str, object]) -> float:
    """
    Get the average popularity of the given items Series
    Args:
        items (pd.DataFrame): a pandas Series that contains string labels ('label')
        pop_by_items (Dict[str, object]): popularity for each label ('label', 'popularity')

    Returns:
        score (float): average popularity
    """
    total_popularity = 0
    for item in items:
        total_popularity += pop_by_items[item]
    return total_popularity / len(items)


def get_avg_pop_by_users(data: pd.DataFrame, pop_by_items: Dict[str, object],
                         group: Set[str] = None) -> Dict[str, float]:
    """
    Get the average popularity for each user in the DataFrame
    Args:
        data (pd.DataFrame): a pandas dataframe with columns = ['from_id', 'to_id', 'rating']
        pop_by_items (Dict[str, object]): popularity for each label ('label', 'popularity')
        group (Set[str]): (optional) the set of users (from_id)

    Returns:
        score_dict (Dict[str, float]): average popularity by user
    """
    if group is None:
        group = data[['from_id']].values.flatten()
    avg_pop_by_users = {}
    for user in group:
        user_items = data.query('from_id == @user')[['to_id']].values.flatten()
        avg_pop_by_users[user] = get_avg_pop(user_items, pop_by_items)

    return avg_pop_by_users


# pop_by_items = Counter(group['item_id'].to_numpy())
# It calculates the Group Average Popularity(GAP)
def calculate_gap(group: Set[str], avg_pop_by_users: Dict[str, object]) -> float:
    """
    Compute the GAP (Group Average Popularity) formula
    Args:
        group (Set[str]): the set of users (from_id)
        avg_pop_by_users (Dict[str, object]): average popularity by user

    Returns:

    """
    total_pop = 0
    #avg_pop_by_users: Dict[str, float] = get_avg_pop_by_users(group=group, data=data, pop_by_items=pop_by_items)
    for element in group:
        total_pop += avg_pop_by_users[element]
    return total_pop / len(group)


def calculate_delta_gap(recs_gap: float, profile_gap: float) -> float:
    """
    Compute the rateo between the recommendation gap and the user profiles gap
    Args:
        recs_gap (float): recommendation gap
        profile_gap: user profiles gap

    Returns:
        score (float): delta gap measure
    """
    return (recs_gap - profile_gap) / profile_gap


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


def split_user_in_groups(score_frame: pd.DataFrame,
                         niche_percentage: float = 0.2,
                         bb_focused_percentage: float = 0.8) -> (Set[str], Set[str], Set[str]):
    """
    Split of DataFrames in 3 different Sets, based on the recommendation popularity of each user
    Args:
        score_frame (pd.DataFrame): DataFrame with columns = ['from_id', 'to_id', 'rating']
        niche_percentage (float):
        bb_focused_percentage (float):

    Returns:
        Tuple(Set[str], Set[str], Set[str]): different sets of 'from_id' divided in three categories
    """

    pop_ratio_by_users = pop_ratio_by_user(score_frame)
    pop_ratio_by_users.sort_values(['popularity'], inplace=True, ascending=False)
    num_of_users = len(pop_ratio_by_users)
    niche_last_index = round(num_of_users * niche_percentage)
    bb_focused_first_index = round(num_of_users * bb_focused_percentage)

    niche_users = set(pop_ratio_by_users['from_id'][:niche_last_index])
    diverse_users = set(pop_ratio_by_users['from_id'][niche_last_index:bb_focused_first_index])
    bb_focused_users = set(pop_ratio_by_users['from_id'][bb_focused_first_index:])

    return niche_users, diverse_users, bb_focused_users

