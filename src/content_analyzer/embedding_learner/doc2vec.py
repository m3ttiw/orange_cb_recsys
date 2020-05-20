from src.content_analyzer.embedding_learner.embedding_learner import Doc2Vec
from src.content_analyzer.information_processor.information_processor import InformationProcessor
from src.content_analyzer.memory_interfaces.memory_interfaces import InformationInterface


class GensimDoc2Vec(Doc2Vec):
    """"
    Class that implements the Abstract Class Word2Vec.
    Implementation of Word2Vec using the Gensim library.
    """
    def __init__(self, loader: InformationInterface, preprocessor: InformationProcessor):
        super().__init__(loader, preprocessor)

    def __str__(self):
        return "GensimDoc2Vec"

    def __repr__(self):
        return "< GensimDoc2Vec :" + \
                "loader = " + str(self.__loader) + \
                "preprocessor = " + str(self.__preprocessor) + " >"

    def start_learning(self):
        """"
        Implementation of the Abstract Method start_training in the Abstract Class Word2vec.
        """
        print("learning")
