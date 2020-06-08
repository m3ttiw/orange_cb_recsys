from pathlib import Path

import pandas as pd


def perform_serendipity(score_frame: pd.DataFrame, algorithm_name: str, out_dir: str = None, num_of_recs: int = 10):
    most_popular_items = set(pd.read_csv('../datasets/most-popular-items.csv').values.flatten())
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
