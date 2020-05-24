from unittest import TestCase

from src.content_analyzer.embedding_learner.random_indexing import RandomIndexing
from src.content_analyzer.raw_information_source import JSONFile
from src.content_analyzer.information_processor.nlp import NLTK


class TestRandomIndexing(TestCase):
    def test_fit(self):
        try:
            RandomIndexing(JSONFile('../../../datasets/movies_info_reduced.json'), NLTK(), ['Genre', 'Plot']).fit()
        except FileNotFoundError:
            RandomIndexing(JSONFile('datasets/movies_info_reduced.json'), NLTK(), ['Genre', 'Plot']).fit()
