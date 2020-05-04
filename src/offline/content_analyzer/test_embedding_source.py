from unittest import TestCase
import numpy as np

from offline.content_analyzer.embedding_source import GensimDownloader


class TestGensimDownloader(TestCase):
    def test_load(self):
        source = GensimDownloader('glove-twitter-25')
        result = source.load("title plot")

        expected = np.ndarray(shape=(2, 25))
        expected[0, :] = [8.50130022e-01, 4.52620000e-01, -7.05750007e-03,
                          -8.77380013e-01, 4.24479991e-01, -8.36589992e-01,
                          8.04159999e-01, 3.74080002e-01, 4.30849999e-01,
                          -6.39360011e-01, 1.19390003e-01, 1.13419998e+00,
                          -3.20650005e+00, 9.31460023e-01, 3.65420014e-01,
                          -3.19309998e-03, 1.97899997e-01, -3.29540014e-01,
                          2.96719998e-01, 4.88689989e-01, -1.37870002e+00,
                          7.52340019e-01, 2.03339994e-01, -6.79979980e-01,
                          -8.91939998e-01]
        expected[1, :] = [7.26029992e-01, 1.46909997e-01, 1.05829999e-01,
                          2.84680009e-01, 2.31950000e-01, -7.86419988e-01,
                          1.33580005e+00, -8.31910014e-01, 4.39669997e-01,
                          -3.01629990e-01, 2.93879986e-01, 4.53700006e-01,
                          -2.18440008e+00, 2.45710000e-01, 3.21599990e-01,
                          6.92149997e-01, 6.65279984e-01, 5.34259975e-01,
                          3.30240000e-03, -4.91389990e-01, -2.80680005e-02,
                          6.41950011e-01, -9.63369980e-02, -9.50479984e-01,
                          -3.88559997e-01]

        self.assertTrue(np.allclose(result, expected))
