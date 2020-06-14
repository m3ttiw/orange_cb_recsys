import os
from collections import Counter

import pandas as pd
from orange_cb_recsys.evaluation.delta_gap import *
from orange_cb_recsys.evaluation.utils import *
import matplotlib.pyplot as plt
from pathlib import Path


# fairness_metrics_results = pd.DataFrame(columns=["user", "gini-index", "delta-gaps", "pop_ratio_profile_vs_recs", "pop_recs_correlation", "recs_long_tail_distr"])
from orange_cb_recsys.utils.const import logger, DEVELOPING, home_path


def perform_gini_index(score_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Gini index score for each user in the DataFrame

    Args:
        score_frame (pd.DataFrame): frame wich stores ('from_id', 'to_id', 'rating')

    Returns:
        results (pd.DataFrame): each row contains ('from_id', 'gini_index')
    """

    logger.info("Computing Gini index")

    def gini(array: np.array):
        """Calculate the Gini coefficient of a numpy array."""

        array = array.flatten()  # all values are treated equally, arrays must be 1d
        array += 1  # values shift, can't be negative
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
                      users_groups: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    Compute the Delta - GAP (Group Average Popularity) metric

    Args:
        truth_frame (pd.DataFrame): frame which stores ('from_id', 'to_id', 'rating') of user profiles
        score_frame (pd.DataFrame): frame which stores ('from_id', 'to_id', 'rating') recommended
        users_groups (Dict[str, Set[str]]): each key contains the name of the group and each value the set of 'from_id'
    Returns:
        results (pd.DataFrame): each row contains ('from_id', 'delta-gap')
    """

    items = score_frame[['to_id']].values.flatten()
    logger.info("Computing pop by items")
    pop_by_items = Counter(items)
    logger.info("Computing recs avg pop by users")
    recs_avg_pop_by_users = get_avg_pop_by_users(score_frame, pop_by_items)

    recommended_users = set(truth_frame[['from_id']].values.flatten())

    score_frame = pd.DataFrame(columns=['user_group', 'delta-gap'])
    for group_name in users_groups:
        logger.info("Computing avg pop by users profiles for delta gap")
        avg_pop_by_users_profiles = get_avg_pop_by_users(truth_frame, pop_by_items, users_groups[group_name])
        logger.info("Computing delta gap for group: %s" % group_name)
        recs_gap = calculate_gap(group=users_groups[group_name].intersection(recommended_users), avg_pop_by_users=recs_avg_pop_by_users)
        profile_gap = calculate_gap(group=users_groups[group_name], avg_pop_by_users=avg_pop_by_users_profiles)
        group_delta_gap = calculate_delta_gap(recs_gap=recs_gap, profile_gap=profile_gap)
        score_frame = score_frame.append(pd.DataFrame({'user_group': [group_name], 'delta-gap': [group_delta_gap]}),
                                         ignore_index=True)
    return score_frame


def perform_pop_ratio_profile_vs_recs(user_groups: Dict[str, Set[str]], truth_frame: pd.DataFrame,
                                      most_popular_items: Set[str], pop_ratio_by_users: pd.DataFrame,
                                      algorithm_name: str, out_dir: str, store_frame: bool = False) -> pd.DataFrame:
    """
    Perform the comparison between the profile popularity and recommendation popularity and build a boxplot

    Args:
        user_groups (Dict[str, Set[str]]): group_name, set of label of 'from_id'
        truth_frame (pd.DataFrame): frame which stores ('from_id', 'to_id', 'rating') of user profiles
        most_popular_items (Set[str]): contains the most popular label of 'to_id'
        pop_ratio_by_users (pd.DataFrame): popularity ratio for each 'from_id'
        algorithm_name (str): name of the algorithm that perform the metric (for plot)
        out_dir (str): directory for saving the plot
        store_frame (bool): store data in the same directory of the boxplot

    Returns:
        score_frame (pd.DataFrame): contains 'user_group', 'profile_pop_ratio', 'recs_pop_ratio'
    """

    logger.info("Computing pop ratio profile vs recs")

    truth_frame = truth_frame[['from_id', 'to_id']]
    score_frame = pd.DataFrame(columns=['user_group', 'profile_pop_ratio', 'recs_pop_ratio'])
    profile_data = []
    recs_data = []
    for group_name in user_groups:
        profile_pop_ratios = get_profile_avg_pop_ratio(user_groups[group_name], pop_ratio_by_users)
        recs_pop_ratios = get_recs_avg_pop_ratio(user_groups[group_name], truth_frame, most_popular_items)
        score_frame = score_frame.append(pd.DataFrame({'user_group': [group_name],
                                                       'profile_pop_ratio': [profile_pop_ratios],
                                                       'recs_pop_ratio': [recs_pop_ratios]}), ignore_index=True)
        profile_data.append(profile_pop_ratios)
        recs_data.append(recs_pop_ratios)

    if store_frame:

        score_frame.to_csv('{}/pop_ratio_profile_vs_recs_{}.csv'.format(out_dir, algorithm_name))

    data_to_plot = [profile_data, recs_data]
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot)

    # Save the figure
    # fig.savefig('fig1.png', bbox_inches='tight')

    ## add patch_artist=True option to ax.boxplot()
    ## to get fill color
    bp = ax.boxplot(data_to_plot, patch_artist=True)

    ## change outline color, fill color and linewidth of the boxes
    for i, box in enumerate(bp['boxes']):
        # change outline color
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='#fcba03')
        if i == 0:
            box.set(facecolor='#1b9e77')

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    ax.set_xticklabels([group_name for group_name in user_groups])

    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.title('{}'.format(algorithm_name))
    plt.ylabel('Ratio of popular items')
    # plt.show()
    # plt.savefig('{}/pop-ratio-profile-vs-recs_{}.svg'.format(out_dir, plot_file_name))

    plt.savefig('{}/pop-ratio-profile-vs-recs_{}.svg'.format(out_dir, algorithm_name))
    plt.clf()

    return score_frame


def perform_pop_recs_correlation(truth_frame: pd.DataFrame, score_frame: pd.DataFrame, algorithm_name: str,
                                 out_dir: str):
    """
    Calculates the correlation between the two frames

    Args:
        truth_frame (pd.DataFrame): frame which stores ('from_id', 'to_id', 'rating') of user profiles
        score_frame (pd.DataFrame): frame which stores ('from_id', 'to_id', 'rating') recommended
        algorithm_name (str): name of the algorithm that perform the metric (for plot)
        out_dir (str): directory for saving the plot
    """

    logger.info("Computing pop recs correlation")

    def build_plot(popularities_, recommendations_, algorithm_name_, out_dir_):
        # Build and save the plot
        plt.scatter(popularities_, recommendations_, marker='o', s=20, c='orange', edgecolors='black', linewidths=0.05)
        plt.title('{}'.format(algorithm_name_))
        plt.xlabel('Popularity')
        plt.ylabel('Recommendation frequency')

        if not DEVELOPING:
            out_dir_ = os.path.join(home_path, 'evaluation_plottings', out_dir_)

        plt.savefig('{}/pop-recs_{}.svg'.format(out_dir_, algorithm_name))
        plt.clf()

    # Calculating popularity by item
    items = truth_frame[['to_id']].values.flatten()
    pop_by_items = Counter(items)

    # Calculating num of recommendations by item
    pop_by_items = pop_by_items.most_common()
    recs_by_item = Counter(score_frame[['to_id']].values.flatten())
    popularities = list()
    recommendations = list()
    popularities_no_zeros = list()
    recommendations_no_zeros = list()

    at_least_one_zero = False
    for item, pop in pop_by_items:
        num_of_recs = recs_by_item[item]

        popularities.append(pop)
        recommendations.append(num_of_recs)

        if num_of_recs != 0:
            popularities_no_zeros.append(pop)
            recommendations_no_zeros.append(num_of_recs)
        else:
            at_least_one_zero = True

    build_plot(popularities, recommendations,
               algorithm_name, out_dir)

    if at_least_one_zero:
        build_plot(popularities_no_zeros, recommendations_no_zeros,
                   algorithm_name + '-no-zeros', out_dir)


def perform_recs_long_tail_distr(truth_frame: pd.DataFrame, algorithm_name: str, out_dir: str):
    """
    Plot the long tail distribution for the truth frame

    Args:
        truth_frame (pd.DataFrame): frame which stores ('from_id', 'to_id', 'rating') of user profiles
        algorithm_name (str): name of the algorithm that perform the metric (for plot)
        out_dir (str): directory for saving the plot
    """
    logger.info("Computing recs long tail distr")

    counts_by_item = Counter(truth_frame[['to_id']].values.flatten())
    ordered_item_count_pairs = counts_by_item.most_common()

    ordered_counts = list()
    for item_count_pair in ordered_item_count_pairs:
        ordered_counts.append(item_count_pair[1])

    plt.plot(ordered_counts)
    plt.title('{}'.format(algorithm_name))
    plt.ylabel('Num of recommendations')
    plt.xlabel('Recommended items')
    # plt.show()

    plt.savefig('{}/recs-long-tail-distr_{}.svg'.format(out_dir, algorithm_name))
    plt.clf()


def catalog_coverage(score_frame: pd.DataFrame, truth_frame: pd.DataFrame) -> float:
    """
    Calculates the catalog coverage

    Args:
        truth_frame (pd.DataFrame): frame which stores ('from_id', 'to_id', 'rating') of user profiles
        score_frame (pd.DataFrame): frame which stores ('from_id', 'to_id', 'rating') recommended

    Returns:
        score (float): coverage percentage
    """
    logger.info("Computing catalog coverage")

    items = set(truth_frame[['to_id']].values.flatten())
    covered_items = set(score_frame[['to_id']].values.flatten())
    coverage_percentage = len(covered_items) / len(items) * 100

    # print('Covered items: ', len(covered_items), ' ({}%)'.format(coverage_percentage))
    return coverage_percentage
