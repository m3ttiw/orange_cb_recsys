from src.offline.content_analyzer.information_processor import InformationProcessor
from src.offline.embedding_learner.embedding_learner import Word2Vec
from src.offline.memory_interfaces.memory_interfaces import InformationInterface


class GensimWord2Vec(Word2Vec):
    """"
    Class that implements the Abstract Class Word2Vec.
    Implementation of Word2Vec using the Gensim library.
    """
    def __init__(self, loader: InformationInterface, preprocessor: InformationProcessor):
        super().__init__(loader, preprocessor)

    def start_learning(self):
        """"
        Implementation of the Abstract Method start_training in the Abstract Class Word2vec.
        """
        print("learning")
