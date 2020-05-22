from unittest import TestCase

from src.content_analyzer.embedding_learner.word2vec import GensimWord2Vec
from src.content_analyzer.information_processor.nlp import NLTK
from src.content_analyzer.raw_information_source import JSONFile
import os


class TestGensimWord2Vec(TestCase):
    def test_fit(self):
        result = GensimWord2Vec(source=JSONFile(os.path.abspath(os.path.join("..", "..", "dataset",
                                "movies_info_reduced.json"))),
                                preprocessor=NLTK(),
                                field_name="Genre").fit()
