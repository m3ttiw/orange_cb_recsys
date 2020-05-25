from unittest import TestCase

from orange_cb_recsys.content_analyzer.embedding_learner.fasttext import GensimFastText
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile


class TestGensimFastText(TestCase):
    def test_fit(self):
        field_list = ['Title', 'Year', 'Genre']
        try:
            GensimFastText(source=JSONFile('../../../datasets/movies_info_reduced.json'),
                           preprocessor=NLTK(),
                           field_list=field_list).fit()
        except FileNotFoundError:
            GensimFastText(source=JSONFile('datasets/movies_info_reduced.json'),
                           preprocessor=NLTK(),
                           field_list=field_list).fit()
