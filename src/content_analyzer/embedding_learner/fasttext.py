from src.content_analyzer.embedding_learner.embedding_learner import FastText
from src.content_analyzer.information_processor.information_processor import InformationProcessor
from src.content_analyzer.memory_interfaces.memory_interfaces import InformationInterface


class GensimFastText(FastText):
    """"
    Class that implements the Abstract Class FastText.
    Implementation of FastText using the Gensim library.
    """

    def __init__(self, loader: InformationInterface, preprocessor: InformationProcessor):
        super().__init__(loader, preprocessor)

    def __str__(self):
        return "FastText"

    def __repr__(self):
        return "< FastText :" + \
               "loader = " + str(self.__loader) + \
               "preprocessor = " + str(self.__preprocessor) + " >"

    def start_learning(self):
        """"
        Implementation of the Abstract Method start_training in the Abstract Class FastText.
        """
        print("learning")
