from unittest import TestCase

from src.content_analyzer.embedding_learner.word2vec import GensimWord2Vec
from src.content_analyzer.information_processor.nlp import NLTK
from src.content_analyzer.raw_information_source import JSONFile
import os


class TestGensimWord2Vec(TestCase):
    def test_fit(self):
        field_list = ['Title', 'Year', 'Genre']
        try:
            GensimWord2Vec(source=JSONFile('../../../datasets/movies_info_reduced.json'),
                           preprocessor=NLTK(),
                           field_list=field_list).fit()
        except FileNotFoundError:
            GensimWord2Vec(source=JSONFile('datasets/movies_info_reduced.json'),
                           preprocessor=NLTK(),
                           field_list=field_list).fit()
