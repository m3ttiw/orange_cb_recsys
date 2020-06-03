from unittest import TestCase

from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsImporter, RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile


class TestRatingsImporter(TestCase):
    def test_import_ratings(self):
        file_path = '../../../datasets/test_import_ratings.json'
        try:
            with open(file_path):
                pass
        except FileNotFoundError:
            file_path = 'datasets/test_import_ratings.json'

        RatingsImporter(source=JSONFile(file_path=file_path),
                        rating_configs=[
                            RatingsFieldConfig(preference_field_name="review_title",
                                               processor=TextBlobSentimentalAnalysis),
                            RatingsFieldConfig()
                        ])
