from unittest import TestCase
from src.content_analyzer.information_processor.nlp import NLTK
from src.content_analyzer.embedding_learner.fasttext import GensimFastText
from src.content_analyzer.raw_information_source import JSONFile


class TestGensimFastText(TestCase):
    def test_fit(self):
            field_list = ['Title', 'Year', 'Genre']
            try:
                    result = GensimFastText(source=JSONFile('../../../datasets/movies_info_reduced.json'),
                                            preprocessor=NLTK(),
                                            field_list=field_list).fit()
            except FileNotFoundError:
                    result = GensimFastText(source=JSONFile('datasets/movies_info_reduced.json'),
                                            preprocessor=NLTK(),
                                            field_list=field_list).fit()
