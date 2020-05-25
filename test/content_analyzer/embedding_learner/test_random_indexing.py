from unittest import TestCase

from orange_cb_recsys.content_analyzer.embedding_learner.random_indexing import RandomIndexing
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile


class TestRandomIndexing(TestCase):
    def test_fit(self):
        try:
            RandomIndexing(JSONFile('../../../datasets/movies_info_reduced.json'), NLTK(), ['Genre', 'Plot']).fit()
        except FileNotFoundError:
            RandomIndexing(JSONFile('datasets/movies_info_reduced.json'), NLTK(), ['Genre', 'Plot']).fit()
