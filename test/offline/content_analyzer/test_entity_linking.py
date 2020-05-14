from unittest import TestCase

from babelpy.babelfy import BabelfyClient
from babelpy.config.config import API_KEY

from src.offline.content_analyzer.entity_linking import BabelPyEntityLinking


class TestBabelPyEntityLinking(TestCase):
    def test_produce_content(self):
        str_ = "text to be babelfyed"
        babelfy_dict = {'bn:00076732n': 0.0}
        content = BabelPyEntityLinking('EN').produce_content("provaEL", str_)
        if content is not None:
            features = content.get_feature_dict()
            for key in features:
                if key in babelfy_dict.keys():
                    self.assertEqual(features[key], babelfy_dict[key], "different global score")
                else:
                    self.fail("{} key not found".format(str(key)))





