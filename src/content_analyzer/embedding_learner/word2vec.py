from typing import List

from src.content_analyzer.embedding_learner import embedding_learner
from src.content_analyzer.information_processor.information_processor import InformationProcessor
from src.content_analyzer.raw_information_source import RawInformationSource


class GensimWord2Vec(embedding_learner.EmbeddingLearner):
    """"
    Class that implements the Abstract Class Word2Vec.
    Implementation of Word2Vec using the Gensim library.
    """

    def __init__(self, source: RawInformationSource,
                 preprocessor: InformationProcessor,
                 field_list=List[str],
                 **kwargs):
        self.__field_list = field_list
        super().__init__(source, preprocessor)

    def __str__(self):
        return "GensimWord2Vec"

    def __repr__(self):
        return "< GensimWord2Vec :" + \
               "source = " + str(self.__source) + \
               "preprocessor = " + str(self.__preprocessor) + " >"

    def fit(self):
        """"
        Implementation of the Abstract Method start_training in the Abstract Class Word2vec.
        """
        print("learning")
