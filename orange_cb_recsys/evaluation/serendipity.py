from pathlib import Path
from typing import Set

import pandas as pd

from orange_cb_recsys.evaluation.utils import popular_items


def perform_serendipity(score_frame: pd.DataFrame, algorithm_name: str, most_popular_items: Set[str] = None,
                        out_dir: str = None, num_of_recs: int = 10) -> float:
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

    print('Serendipity: {}'.format(serendipity))

    # Serializing results
    try:
        file_path = '{}/serendipity_{}.csv'.format(out_dir, algorithm_name)
        with open(file_path, 'a', newline='') as f:
            f.write("%s,%f\n" % ('serendipity: ', serendipity))
    except FileNotFoundError:
        try:
            file_path = '../../{}/serendipity_{}.csv'.format(out_dir, algorithm_name)
            with open(file_path, 'a', newline='') as f:
                f.write("%s,%f\n" % ('serendipity: ', serendipity))
        except FileNotFoundError:
            file_path = '{}/serendipity_{}.csv'.format(str(Path.home()), algorithm_name)
            with open(file_path, 'a', newline='') as f:
                f.write("%s,%f\n" % ('serendipity: ', serendipity))
    print('saved in: {}'.format(file_path))

    return serendipity
