from unittest import TestCase

from babelpy.babelfy import BabelfyClient
from babelpy.config.config import API_KEY

from src.offline.content_analyzer.entity_linking import BabelPyEntityLinking


class TestBabelPyEntityLinking(TestCase):
    def test_produce_content(self):
        str_ = "text to be babelfyed"
        babelfy_dict = {"sysnetID": "global score"}  # TO DO
        content = BabelPyEntityLinking('EN').produce_content(str_)
        if content is not None:
            self.skipTest("Get non yet implemented")
            features = content.get_features()
            for key in features:
                if key in babelfy_dict.keys():
                    self.assertEqual(features[key], babelfy_dict[key], "different global score")
                else:
                    self.fail("{} key not found".format(str(key)))





