import os

import pandas as pd

from orange_cb_recsys.utils.const import home_path, DEVELOPING


def load_ratings(filename: str):
    if not DEVELOPING:
        return pd.read_csv(os.path.join(home_path, filename), dtype=str)
    else:
        return pd.read_csv(filename, dtype=str)
