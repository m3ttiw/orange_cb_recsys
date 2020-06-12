from typing import List

from gensim.corpora import Dictionary
from gensim.models import LsiModel

from orange_cb_recsys.content_analyzer.embedding_learner.embedding_learner import EmbeddingLearner
from orange_cb_recsys.content_analyzer.information_processor.information_processor import TextProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource


class GensimLatentSemanticAnalysis(EmbeddingLearner):
    """
    Class that implements latent semantic analysis using Gensim
    """
    def __init__(self, source: RawInformationSource, preprocessor: TextProcessor, field_list: List[str]):
        super().__init__(source, preprocessor, field_list)

    def __str__(self):
        return "GensimLatentSemanticAnalysis"

    def __repr__(self):
        return "< GensimLatentSemanticAnalysis :" + \
                "source = " + str(self.__source) + \
                "preprocessor = " + str(self.__preprocessor) + " >"

    @staticmethod
    def __create_dictionary(docs) -> Dictionary:
        return Dictionary(docs)

    @staticmethod
    def __create_word_docs_matrix(docs, dictionary) -> List[str]:
        """
        Returns:
             docs-words matrix, that contains a mapping between the IDs and the words
        """
        return [dictionary.doc2bow(doc) for doc in docs]

    def fit(self):
        """
        Creates the model for the embedding
        """
        docs = self.extract_corpus()
        dictionary = GensimLatentSemanticAnalysis.__create_dictionary(docs)
        word_docs_matrix = GensimLatentSemanticAnalysis.__create_word_docs_matrix(docs, dictionary)
        self.set_model(LsiModel(word_docs_matrix, id2word=dictionary))