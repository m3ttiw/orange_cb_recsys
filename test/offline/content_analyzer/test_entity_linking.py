from unittest import TestCase

from babelpy.babelfy import BabelfyClient
from babelpy.config.config import API_KEY

from src.offline.content_analyzer.entity_linking import BabelPyEntityLinking


class TestBabelPyEntityLinking(TestCase):
    def test_produce_content(self):
        str_ = "text to be babelfyed"
        content = BabelPyEntityLinking('EN').produce_content(str_)
        if content is not None:
            self.assertIn()




