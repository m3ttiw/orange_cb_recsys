from abc import abstractmethod

import pandas as pd
from orange_cb_recsys.evaluation.delta_gap import *
from orange_cb_recsys.evaluation.metrics import Metric
from orange_cb_recsys.evaluation.utils import *
import matplotlib.pyplot as plt
from pathlib import Path

from orange_cb_recsys.utils.const import logger


class FairnessMetric(Metric):
    def __init__(self, algorithm_name: str, out_dir: str):
        self.__algorithm_name = algorithm_name
        self.__out_dir = out_dir

    def get_algorithm_name(self):
        return self.__algorithm_name

    def get_output_directory(self):
        return self.__out_dir

    @abstractmethod
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        raise NotImplementedError


class GroupFairnessMetric(FairnessMetric):
    def __init__(self, algorithm_name: str, out_dir: str, user_groups: Dict[str, float]):
        super().__init__(algorithm_name, out_dir)
        self.__user_groups = user_groups

    def get_user_groups(self):
        return self.__user_groups

    @abstractmethod
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        raise NotImplementedError


class GiniIndex(Metric):
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame = None):
        """
        Calculate Gini index score for each user in the DataFrame

        Args:
            truth:
            predictions (pd.DataFrame): frame wich stores ('from_id', 'to_id', 'rating')

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
        for idx, df in predictions.groupby('from_id'):
            # from the DataFrame exract the 'rating' column as a np.array and create a Gini_index
            # store the score in a dict
            score_dict[df['from_id'].iloc[0]] = gini(df['rating'].to_numpy())
        results = pd.DataFrame({'from': list(score_dict.keys()), 'gini-index': list(score_dict.values())})

        return results


class PopRecsCorrelation(FairnessMetric):
    def __init__(self, algorithm_name: str, out_dir: str):
        super().__init__(algorithm_name, out_dir)

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Calculates the correlation between the two frames

        Args:
        """

        logger.info("Computing pop recs correlation")

        def build_plot(popularities_, recommendations_, algorithm_name_, out_dir_):
            # Build and save the plot
            plt.scatter(popularities_, recommendations_, marker='o', s=20, c='orange', edgecolors='black', linewidths=0.05)
            plt.title('{}'.format(algorithm_name_))
            plt.xlabel('Popularity')
            plt.ylabel('Recommendation frequency')
            try:
                plt.savefig('{}/pop-recs_{}.svg'.format(out_dir_, algorithm_name_))
            except FileNotFoundError:
                try:
                    plt.savefig('../../{}/pop-recs_{}.svg'.format(out_dir_, algorithm_name_))
                except FileNotFoundError:
                    plt.savefig('{}/pop-recs_{}.svg'.format(str(Path.home()), algorithm_name_))
            plt.clf()

        # Calculating popularity by item
        items = truth[['to_id']].values.flatten()
        pop_by_items = Counter(items)

        # Calculating num of recommendations by item
        pop_by_items = pop_by_items.most_common()
        recs_by_item = Counter(predictions[['to_id']].values.flatten())
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
                   self.get_algorithm_name(), self.get_output_directory())

        if at_least_one_zero:
            build_plot(popularities_no_zeros, recommendations_no_zeros,
                       self.get_algorithm_name() + '-no-zeros', self.get_output_directory())


class LongTailDistr(FairnessMetric):
    def __init__(self, algorithm_name: str, out_dir: str):
        super().__init__(algorithm_name, out_dir)

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Plot the long tail distribution for the truth frame

        Args:

        """
        logger.info("Computing recs long tail distr")

        counts_by_item = Counter(truth[['to_id']].values.flatten())
        ordered_item_count_pairs = counts_by_item.most_common()

        ordered_counts = list()
        for item_count_pair in ordered_item_count_pairs:
            ordered_counts.append(item_count_pair[1])

        plt.plot(ordered_counts)
        plt.title('{}'.format(self.get_algorithm_name()))
        plt.ylabel('Num of recommendations')
        plt.xlabel('Recommended items')
        # plt.show()
        try:
            plt.savefig('{}/recs-long-tail-distr_{}.svg'.format(self.get_output_directory(),
                                                                self.get_algorithm_name()))
        except FileNotFoundError:
            try:
                plt.savefig('../../{}/recs-long-tail-distr_{}.svg'.format(self.get_output_directory(),
                                                                          self.get_algorithm_name()))
            except FileNotFoundError:
                plt.savefig('{}/recs-long-tail-distr_{}.svg'.format(str(Path.home()),
                                                                    self.get_algorithm_name()))

        plt.clf()


class CatalogCoverage(FairnessMetric):
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Calculates the catalog coverage

        Returns:
            score (float): coverage percentage
        """
        logger.info("Computing catalog coverage")

        items = set(truth[['to_id']].values.flatten())
        covered_items = set(predictions[['to_id']].values.flatten())
        coverage_percentage = len(covered_items) / len(items) * 100

        return coverage_percentage


class DeltaGap(GroupFairnessMetric):
    def __init__(self, user_groups: Dict[str, float]):
        super().__init__(None, None, user_groups)

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Compute the Delta - GAP (Group Average Popularity) metric

        Returns:
            results (pd.DataFrame): each row contains ('from_id', 'delta-gap')
        """

        pop_items = popular_items(score_frame=truth)
        user_groups = split_user_in_groups(score_frame=predictions, groups=self.get_user_groups(), pop_items=pop_items)
        items = predictions[['to_id']].values.flatten()
        logger.info("Computing pop by items")
        pop_by_items = Counter(items)
        logger.info("Computing recs avg pop by users")
        recs_avg_pop_by_users = get_avg_pop_by_users(predictions, pop_by_items)

        recommended_users = set(truth[['from_id']].values.flatten())

        score_frame = pd.DataFrame(columns=['user_group', 'delta-gap'])
        for group_name in user_groups:
            logger.info("Computing avg pop by users profiles for delta gap")
            avg_pop_by_users_profiles = get_avg_pop_by_users(truth, pop_by_items, user_groups[group_name])
            logger.info("Computing delta gap for group: %s" % group_name)
            recs_gap = calculate_gap(group=user_groups[group_name].intersection(recommended_users),
                                     avg_pop_by_users=recs_avg_pop_by_users)
            profile_gap = calculate_gap(group=user_groups[group_name], avg_pop_by_users=avg_pop_by_users_profiles)
            group_delta_gap = calculate_delta_gap(recs_gap=recs_gap, profile_gap=profile_gap)
            score_frame = score_frame.append(pd.DataFrame({'user_group': [group_name], 'delta-gap': [group_delta_gap]}),
                                             ignore_index=True)
        return score_frame


class PopRatioVsRecs(GroupFairnessMetric):
    def __init__(self, algorithm_name: str,
                 out_dir: str, user_groups: Dict[str, float],
                 store_frame: bool):
        super().__init__(algorithm_name, out_dir, user_groups)
        self.__user_groups = user_groups
        self.__store_frame = store_frame

    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Perform the comparison between the profile popularity and recommendation popularity and build a boxplot

        Args:
            predictions:
            truth (pd.DataFrame): frame which stores ('from_id', 'to_id', 'rating') of user profiles

        Returns:
            score_frame (pd.DataFrame): contains 'user_group', 'profile_pop_ratio', 'recs_pop_ratio'
        """

        logger.info("Computing pop ratio profile vs recs")
        most_popular_items = popular_items(score_frame=truth)
        pop_ratio_by_users = pop_ratio_by_user(score_frame=predictions, most_pop_items=most_popular_items)

        pop_items = popular_items(score_frame=truth)
        user_groups = split_user_in_groups(score_frame=predictions, groups=self.get_user_groups(), pop_items=pop_items)

        truth = truth[['from_id', 'to_id']]
        score_frame = pd.DataFrame(columns=['user_group', 'profile_pop_ratio', 'recs_pop_ratio'])
        profile_data = []
        recs_data = []
        for group_name in user_groups:
            profile_pop_ratios = get_profile_avg_pop_ratio(user_groups[group_name], pop_ratio_by_users)
            recs_pop_ratios = get_recs_avg_pop_ratio(user_groups[group_name], truth, most_popular_items)
            score_frame = score_frame.append(pd.DataFrame({'user_group': [group_name],
                                                           'profile_pop_ratio': [profile_pop_ratios],
                                                           'recs_pop_ratio': [recs_pop_ratios]}), ignore_index=True)
            profile_data.append(profile_pop_ratios)
            recs_data.append(recs_pop_ratios)

        if self.__store_frame:
            try:
                score_frame.to_csv('{}/pop_ratio_profile_vs_recs_{}.csv'.format(self.get_output_directory(),
                                                                                self.get_algorithm_name()))
            except FileNotFoundError:
                try:
                    score_frame.to_csv('../../{}/pop_ratio_profile_vs_recs_{}.csv'.format(self.get_output_directory(),
                                                                                          self.get_algorithm_name()))
                except FileNotFoundError:
                    score_frame.to_csv(
                        '{}/pop_ratio_profile_vs_recs_{}.csv'.format(str(Path.home()), self.get_algorithm_name()))

        data_to_plot = [profile_data, recs_data]
        # Create a figure instance
        fig = plt.figure(1, figsize=(9, 6))

        # Create an axes instance
        ax = fig.add_subplot(111)

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

        ax.set_xticklabels([group_name for group_name in self.__user_groups])

        ## Remove top axes and right axes ticks
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.title('{}'.format(self.__algorithm_name))
        plt.ylabel('Ratio of popular items')
        try:
            plt.savefig('{}/pop-ratio-profile-vs-recs_{}.svg'.format(self.get_output_directory(),
                                                                     self.get_algorithm_name()))
        except FileNotFoundError:
            try:
                plt.savefig('../../{}/pop-ratio-profile-vs-recs_{}.svg'.format(self.get_output_directory(),
                                                                               self.get_algorithm_name()))
            except FileNotFoundError:
                plt.savefig('{}/pop-ratio-profile-vs-recs_{}.svg'.format(str(Path.home()),
                                                                         self.get_algorithm_name()))

        plt.clf()

        return score_frame
