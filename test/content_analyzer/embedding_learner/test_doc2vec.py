from unittest import TestCase
import os
from content_analyzer.embedding_learner.doc2vec import GensimDoc2Vec
from content_analyzer.information_processor.nlp import NLTK
from src.content_analyzer.raw_information_source import JSONFile


class TestGensimDoc2Vec(TestCase):
    def test_start_learning(self):
        test_list = [[0.11635043, 0.28827286, 0.26367947, -0.0368935, -0.07216292, -0.12885164, -0.06062358, 0.290583,
                      0.6601088, 0.2113433, -0.61001587, 0.1947201, -0.17322248, -0.44282833, 0.4947037, -0.41412833,
                      -0.91901696, -0.00481507, 0.21716033, 0.09160328],
                     [0.08266724, 0.18558007, 0.17412613, -0.02734668, -0.05606755, -0.05064456, -0.04156437,
                      0.12810938, 0.4672991, 0.14360814, -0.44134668, 0.1255639, -0.06440181, -0.23697469, 0.3259466,
                      -0.298236, -0.6237428, -0.04468865, 0.1474588, 0.08139123],
                     [0.03376862, 0.13182545, 0.1217874, -0.0171805, -0.06975423, -0.04341289, -0.04310595, 0.13309316,
                      0.38239855, 0.11585647, -0.38156882, 0.15640694, -0.10280843, -0.2742574, 0.25440186, -0.18735062,
                      -0.550365, -0.04548627, 0.09908516, 0.09024432],
                     [0.05343575, 0.20417786, 0.19907168, -0.05516035, -0.03117811, -0.12420024, -0.01822128,
                      0.24684472, 0.552818, 0.10604687, -0.47952488, 0.18031257, -0.14340155, -0.370736, 0.3842013,
                      -0.34952304, -0.70719564, -0.00113534, 0.09173155, 0.05633677]]
        try:
            test_results = GensimDoc2Vec(source=JSONFile(file_path="datasets/d2v_test_data.json"),
                                         preprocessor=NLTK(), field_name="doc_field").start_learning()
        except FileNotFoundError:
            test_results = GensimDoc2Vec(source=JSONFile(file_path="../../../datasets/d2v_test_data.json"),
                                         preprocessor=NLTK(), field_name="doc_field").start_learning()

        for i, res in enumerate(test_results):
            self.assertEqual(test_list[i], res, "Fail in Doc {} - Vector = {}".format(str(i), res))
