from typing import List

from gensim.corpora import Dictionary
from gensim.models import LsiModel

from src.content_analyzer.embedding_learner.embedding_learner import EmbeddingLearner
from src.content_analyzer.information_processor.information_processor import InformationProcessor
from src.content_analyzer.raw_information_source import RawInformationSource


class GensimLatentSemanticAnalysis(EmbeddingLearner):
    """
    Class that implements latent semantic analysis using Gensim
    """
    def __init__(self, source: RawInformationSource, preprocessor: InformationProcessor, field_list: List[str]):
        super().__init__(source, preprocessor, field_list)

    def __str__(self):
        return "GensimLatentSemanticAnalysis"

    def __repr__(self):
        return "< GensimLatentSemanticAnalysis :" + \
                "source = " + str(self.__source) + \
                "preprocessor = " + str(self.__preprocessor) + " >"

    def load_data_from_source(self) -> List[List[str]]:
        my_iter = iter(self.get_source())
        end_loop = False
        docs = []
        while not end_loop:
            try:
                row = next(my_iter)
            except StopIteration:
                end_loop = True
            else:
                text = ""
                for field_name in self.get_field_list():
                    text += " " + row[field_name]
                text = self.get_preprocessor().process(text)
                docs.append(text)
        return docs

    @staticmethod
    def create_dictionary(docs) -> Dictionary:
        return Dictionary(docs)

    @staticmethod
    def create_word_docs_matrix(docs, dictionary) -> List[str]:
        return [dictionary.doc2bow(doc) for doc in docs]

    def fit(self, dictionary, word_docs_matrix):
        """
        Creates the model for the embedding

        Args:
            dictionary: Can be either a stream of document vectors or a sparse matrix of shape
                (number_of_terms, number_of_documents)
            word_docs_matrix: Mapping between the IDs and the words

        Returns:
            LsiModel: The LSI (Latent Semantic Index) built.
        """
        return LsiModel(word_docs_matrix, id2word=dictionary)

    @staticmethod
    def save_model(name, model: LsiModel):
        name = name+".model"
        model.save(name)
