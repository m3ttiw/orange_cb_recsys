from unittest import TestCase

from orange_cb_recsys.content_analyzer.field_content_production_techniques.sentimental_analysis import \
    TextBlobSentimentalAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile


class TestTextBlobSentimentalAnalysis(TestCase):
    def test_calculate_score(self):
        file_path = '../../../datasets/test_sentiment_analysis.json.json'
        try:
            with open(file_path):
                pass
        except FileNotFoundError:
            file_path = 'datasets/test_sentiment_analysis.json'

        confront_list = [1.0, 0.9, -0.6999999999999998]
        test_list = TextBlobSentimentalAnalysis(source=JSONFile(file_path), field_name="rating")
        self.assertEqual(test_list, confront_list)
