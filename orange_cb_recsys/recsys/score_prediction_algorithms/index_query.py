import os

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.content_analyzer.memory_interfaces import IndexInterface
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import ScorePredictionAlgorithm

import pandas as pd

from orange_cb_recsys.utils.const import DEVELOPING, home_path


class IndexQuery(ScorePredictionAlgorithm):
    def get_query(self):
        pass

    def predict(self, item: Content, ratings: pd.DataFrame, items_directory: str):
        index_path = os.path.join(items_directory, 'search_index')
        if not DEVELOPING:
            index_path = os.path.join(home_path, items_directory, 'search_index')

        index_interface = IndexInterface(index_path)

        for item_id in ra