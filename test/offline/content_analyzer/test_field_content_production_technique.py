from unittest import TestCase

from offline.content_analyzer.combining_technique import Centroid
from offline.content_analyzer.embedding_source import GensimDownloader
from offline.content_analyzer.field_content_production_technique import EmbeddingTechnique, Granularity


class TestEmbeddingTechnique(TestCase):
    def test_produce_content(self):
        technique = EmbeddingTechnique(Centroid(), GensimDownloader('word2vec-google-news-300'), Granularity.DOC)

        technique.produce_content("title plot")
