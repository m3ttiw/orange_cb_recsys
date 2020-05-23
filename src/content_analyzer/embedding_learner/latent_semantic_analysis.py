from typing import List

from gensim.corpora import Dictionary
from gensim.models import LsiModel

from src.content_analyzer.embedding_learner.embedding_learner import EmbeddingLearner
from src.content_analyzer.information_processor.information_processor import TextProcessor
from src.content_analyzer.raw_information_source import RawInformationSource


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
    def create_dictionary(docs) -> Dictionary:
        return Dictionary(docs)

    @staticmethod
    def create_word_docs_matrix(docs, dictionary) -> List[str]:
        return [dictionary.doc2bow(doc) for doc in docs]

    def fit(self):
        """
        Creates the model for the embedding

        Args:
            dictionary: Can be either a stream of document vectors or a sparse matrix of shape
                (number_of_terms, number_of_documents)
            word_docs_matrix: Mapping between the IDs and the words

        Returns:
            LsiModel: The LSI (Latent Semantic Index) built.
        """
        docs = self.extract_corpus()
        dictionary = GensimLatentSemanticAnalysis.create_dictionary(docs)
        word_docs_matrix = GensimLatentSemanticAnalysis.create_word_docs_matrix(docs, dictionary)
        return LsiModel(word_docs_matrix, id2word=dictionary)

