from unittest import TestCase

from orange_cb_recsys.content_analyzer.embedding_learner.latent_semantic_analysis import GensimLatentSemanticAnalysis
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile


class TestGensimLatentSemanticAnalysis(TestCase):
    def test_fit(self):
        preprocessor = NLTK(stopwords_removal=True)
        fields = ["Plot"]
        try:
            src = JSONFile("datasets/movies_info_reduced.json")
            learner = GensimLatentSemanticAnalysis(src, preprocessor, fields)
            learner.fit()
        except FileNotFoundError:
            src = JSONFile("../../../datasets/movies_info_reduced.json")
            learner = GensimLatentSemanticAnalysis(src, preprocessor, fields)
            learner.fit()
