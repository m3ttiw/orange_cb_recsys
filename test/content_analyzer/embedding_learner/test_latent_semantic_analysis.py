from unittest import TestCase
from src.content_analyzer.embedding_learner.latent_semantic_analysis import GensimLatentSemanticAnalysis
from src.content_analyzer.information_processor.nlp import NLTK
from src.content_analyzer.raw_information_source import JSONFile
import os


class TestGensimLatentSemanticAnalysis(TestCase):
    os.chdir("../../../datasets")
    def test_load_data_from_source(self):
        src = JSONFile("movies_info_reduced.json")
        preprocessor = NLTK(stopwords_removal=True, stemming=True)
        learner = GensimLatentSemanticAnalysis(src, preprocessor, ["Title", "Released"])
        expected = [['jumanji', '15', 'dec', '1995'],
                    ['grumpier', 'old', 'men', '22', 'dec', '1995'],
                    ['toy', 'stori', '22', 'nov', '1995'],
                    ['father', 'bride', 'part', 'ii', '08', 'dec', '1995'],
                    ['heat', '15', 'dec', '1995'],
                    ['tom', 'huck', '22', 'dec', '1995'],
                    ['wait', 'exhal', '22', 'dec', '1995'],
                    ['sabrina', '15', 'dec', '1995'],
                    ['dracula', 'dead', 'love', 'it', '22', 'dec', '1995'],
                    ['nixon', '05', 'jan', '1996'],
                    ['the', 'american', 'presid', '17', 'nov', '1995'],
                    ['goldeney', '17', 'nov', '1995'],
                    ['balto', '22', 'dec', '1995'],
                    ['cutthroat', 'island', '22', 'dec', '1995'],
                    ['casino', '22', 'nov', '1995'],
                    ['sudden', 'death', '22', 'dec', '1995'],
                    ['sens', 'sensibl', '26', 'jan', '1996'],
                    ['four', 'room', '25', 'dec', '1995'],
                    ['money', 'train', '22', 'nov', '1995'],
                    ['ace', 'ventura', 'when', 'natur', 'call', '10', 'nov', '1995']]
        self.assertEqual(learner.load_data_from_source(),
                         expected)

    def test_create_dictionary(self):
        src = JSONFile("movies_info_reduced.json")
        preprocessor = NLTK(stopwords_removal=True, stemming=True)
        learner = GensimLatentSemanticAnalysis(src, preprocessor, ["Title", "Released"])
        docs = learner.load_data_from_source()
        self.assertEqual(len(learner.create_dictionary(docs)), 55)

    def test_create_word_docs_matrix(self):
        src = JSONFile("movies_info_test.json")
        preprocessor = NLTK(stopwords_removal=True)
        learner = GensimLatentSemanticAnalysis(src, preprocessor, ["Plot"])
        docs = learner.load_data_from_source()
        dct = learner.create_dictionary(docs)
        expected = [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 2),
                     (12, 2), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1)],
                    [(22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 2), (30, 3), (31, 1), (32, 1),
                     (33, 1), (34, 1), (35, 1), (36, 1), (37, 1), (38, 1), (39, 1), (40, 1), (41, 1), (42, 2), (43, 1),
                     (44, 1), (45, 1), (46, 1), (47, 1), (48, 1), (49, 1), (50, 1), (51, 1), (52, 1), (53, 1), (54, 2),
                     (55, 1), (56, 2), (57, 1), (58, 1), (59, 1), (60, 2), (61, 1), (62, 1), (63, 1), (64, 1), (65, 1),
                     (66, 1), (67, 1), (68, 1), (69, 1), (70, 1), (71, 2), (72, 1), (73, 1), (74, 1), (75, 1)]]
        self.assertEqual(learner.create_word_docs_matrix(docs, dct),
                         expected)

    def test_fit(self):
        src = JSONFile("movies_info_reduced.json")
        preprocessor = NLTK(stopwords_removal=True)
        learner = GensimLatentSemanticAnalysis(src, preprocessor, ["Plot"])
        docs = learner.load_data_from_source()
        dct = learner.create_dictionary(docs)
        word_docs_matrix = learner.create_word_docs_matrix(docs, dct)
        learner.fit(dct, word_docs_matrix)

    def test_save(self):
        src = JSONFile("movies_info_reduced.json")
        preprocessor = NLTK(stopwords_removal=True)
        learner = GensimLatentSemanticAnalysis(src, preprocessor, ["Plot"])
        docs = learner.load_data_from_source()
        dct = learner.create_dictionary(docs)
        word_docs_matrix = learner.create_word_docs_matrix(docs, dct)
        model = learner.fit(dct, word_docs_matrix)
        learner.save_model("gensim_lsa", model)
        self.assertIn("gensim_lsa.model", os.listdir())
