from unittest import TestCase
from gensim.models.doc2vec import Doc2Vec
from src.content_analyzer.embedding_learner.doc2vec import GensimDoc2Vec
from src.content_analyzer.information_processor.nlp import NLTK
from src.content_analyzer.raw_information_source import JSONFile


class TestGensimDoc2Vec(TestCase):
    def test_fit(self):

        try:
            path = "datasets/d2v_test_data.json"
            open(path)
        except FileNotFoundError:
            path = "../../../datasets/d2v_test_data.json"

        GensimDoc2Vec(source=JSONFile(file_path=path), preprocessor=NLTK(), field_list=["doc_field"]).fit()
