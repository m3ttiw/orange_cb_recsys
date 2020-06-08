import math
from collections import Counter
from pathlib import Path

import pandas as pd


def perform_novelty(score_frame: pd.Dataframe, truth_frame: pd.DataFrame,
                    algorithm_name: str, out_dir: str = None, num_of_recs: int = 10) -> float:
    total_ratings = len(truth_frame.index)
    ratings_by_item = Counter(truth_frame[['to_id']].values.flatten())
    users = set(score_frame[['from_id']].values.flatten())

    users_log_popularity = 0
    for user in users:
        user_recs = set(score_frame.query('from_id == @user')[['to_id']].values.flatten())
        user_log_popularity = 0
        for item in user_recs:
            item_pop = (ratings_by_item[item] + 1) / total_ratings
            user_log_popularity += math.log2(item_pop)
        users_log_popularity += user_log_popularity

    novelty = - (users_log_popularity / (len(users) * num_of_recs))

    print('Novelty: {}'.format(novelty))
    try:
        file_path = '{}/novelty_{}.csv'.format(out_dir, algorithm_name)
        with open(file_path, 'a', newline='') as f:
            f.write("%s,%f\n" % ('novelty: ', novelty))
    except FileNotFoundError:
        try:
            file_path = '../../{}/novelty_{}.csv'.format(out_dir, algorithm_name)
            with open(file_path, 'a', newline='') as f:
                f.write("%s,%f\n" % ('novelty: ', novelty))
        except FileNotFoundError:
            file_path = '{}/novelty_{}.csv'.format(str(Path.home()), algorithm_name)
            with open(file_path, 'a', newline='') as f:
                f.write("%s,%f\n" % ('novelty: ', novelty))
    print('saved in: {}'.format(file_path))
    return novelty
