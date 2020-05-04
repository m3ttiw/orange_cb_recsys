from unittest import TestCase
import numpy as np
from offline.content_analyzer.combining_technique import Centroid
from offline.content_analyzer.embedding_source import BinaryFile, GensimDownloader
from offline.content_analyzer.field_content_production_technique import EmbeddingTechnique, Granularity


class TestEmbeddingTechnique(TestCase):
    def test_produce_content(self):
        technique = EmbeddingTechnique(Centroid(), GensimDownloader('glove-twitter-25'), Granularity.DOC)

        result = technique.produce_content("title plot")

        expected = np.ndarray(shape=(25,))
        expected[:] = [7.88080007e-01, 2.99764998e-01, 4.93862494e-02, -2.96350002e-01,
                       3.28214996e-01, -8.11504990e-01, 1.06998003e+00, -2.28915006e-01,
                       4.35259998e-01, -4.70495000e-01, 2.06634995e-01, 7.93949991e-01,
                       -2.69545007e+00, 5.88585012e-01, 3.43510002e-01, 3.44478448e-01,
                       4.31589991e-01, 1.02359980e-01, 1.50011199e-01, -1.35000050e-03,
                       -7.03384009e-01, 6.97145015e-01, 5.35014980e-02, -8.15229982e-01,
                       -6.40249997e-01]
        print(result)

        self.assertTrue(np.allclose(result, expected))
