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
        score (float): gap score
    """
    total_pop = 0
    #avg_pop_by_users: Dict[str, float] = get_avg_pop_by_users(group=group, data=data, pop_by_items=pop_by_items)
    for element in group:
        try:
            total_pop += avg_pop_by_users[element]
        except KeyError:
            pass
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




