from abc import ABC


class InformationProcessor(ABC):
    def __init__(self):
        pass


class ImageProcessor(InformationProcessor):
    """
    Future Feature
    """
    def __init__(self):
        super().__init__()


class AudioProcessor(InformationProcessor):
    """
    Future Feature
    """
    def __init__(self):
        super().__init__()


class TextProcessor(InformationProcessor):
    def __init__(self):
        super().__init__()

    def deserialize(self):
        pass


class NLP(TextProcessor, ABC):
    def __init__(self, stopwords_removal: bool = False,
                 stemming: bool = False,
                 lemmatization: bool = False,
                 named_entity_recognition: bool = False,
                 strip_multiple_whitespaces: bool = True):
        super().__init__()
        self.__stopwords_removal = stopwords_removal
        self.__stemming = stemming
        self.__lemmatization = lemmatization
        self.__named_entity_recognition = named_entity_recognition
        self.__strip_multiple_whitespaces = strip_multiple_whitespaces
