from unittest import TestCase
import numpy as np

from offline.content_analyzer.combining_technique import Centroid


class TestCentroid(TestCase):
    def test_combine(self):
        z = np.ndarray(shape=(3, 3))

        z[0, :] = [1, 1, 1]
        z[1, :] = [2, 2, 2]
        z[2, :] = [3, 3, 3]

        combiner = Centroid()
        result = combiner.combine(z)

        expected = np.ndarray(shape=(3, ))
        expected[:] = [2, 2, 2]

        self.assertTrue(np.array_equal(result, expected))
