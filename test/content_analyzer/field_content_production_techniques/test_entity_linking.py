from unittest import TestCase

from orange_cb_recsys.content_analyzer.field_content_production_techniques.entity_linking import BabelPyEntityLinking


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





