from unittest import TestCase
import pandas as pd
import numpy as np
from orange_cb_recsys.evaluation.eval_model import EvalModel, RankingMetricsConfig, FairnessMetricsConfig
from orange_cb_recsys.evaluation.partitioning import KFoldPartitioning
from orange_cb_recsys.recsys import CosineSimilarity
from orange_cb_recsys.recsys.config import RecSysConfig
from orange_cb_recsys.recsys.ranking_algorithms.centroid_vector import CentroidVector


class TestEvalModel(TestCase):
    def test_fit(self):
        item_id_list = [
            'tt0112281',
            'tt0112302',
            'tt0112346',
            'tt0112453',
            'tt0112641',
            'tt0112760',
            'tt0112896',
            'tt0113041',
            'tt0113101',
            'tt0113189',
            'tt0113228',
            'tt0113277',
            'tt0113497',
            'tt0113845',
            'tt0113987',
            'tt0114319',
            'tt0114388',
            'tt0114576',
            'tt0114709',
            'tt0114885',
        ]
        record_list = []
        for i in range(1, 7):
            extract_items = set([x for i, x in enumerate(item_id_list) if np.random.randint(0, 2) == 1 and i < 10])
            for item in extract_items:
                record_list.append((str(i), item, str(np.random.randint(-10, 11) / 10)))
        t_ratings = pd.DataFrame.from_records(record_list, columns=['from_id', 'to_id', 'score'])
        recsys_config = RecSysConfig(
            users_directory='contents/users_test1591814865.8959296',
            items_directory='contents/movielens_test1591885241.5520566',
            score_prediction_algorithm=None,
            ranking_algorithm=CentroidVector(
                item_field='Plot',
                field_representation='1',
                similarity=CosineSimilarity()
            ),
            rating_frame=t_ratings
        )
        ranking_config = RankingMetricsConfig(
            relevant_threshold=0.0,
            relevance_split={0: (-1.0, 0.0), 1: (0.0, 0.3), 2: (0.3, 0.7), 3: (0.7, 1.0)}
        )
        fairness_config = FairnessMetricsConfig(
            output_directory='datasets',
            user_groups={'a': 0.2, 'b': 0.4}
        )
        EvalModel(config=recsys_config,
                  partitioning=KFoldPartitioning(),
                  prediction_metric=False,
                  ranking_metrics_config=ranking_config,
                  fairness_metric_config=fairness_config,
                  serendipity_novelty_metrics=True
                  ).fit()
