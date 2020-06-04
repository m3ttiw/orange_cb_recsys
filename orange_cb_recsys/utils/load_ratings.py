import os

import pandas as pd

from orange_cb_recsys.utils.const import home_path


def load_ratings(filename: str):
    return pd.read_csv(os.path.join(home_path, filename))
