from abc import ABC, abstractmethod


class InformationProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, field_data):
        pass


class ImageProcessor(InformationProcessor):
    """
    Future Feature
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def process(self, field_data):
        pass


class AudioProcessor(InformationProcessor):
    """
    Future Feature
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def process(self, field_data):
        pass


class TextProcessor(InformationProcessor):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def process(self, field_data):
        pass


class NLP(TextProcessor):
    def __init__(self, stopwords_removal: bool = False,
                 stemming: bool = False,
                 lemmatization: bool = False,
                 named_entity_recognition: bool = False,
                 strip_multiple_whitespaces: bool = True):
        super().__init__()
        self.__stopwords_removal: bool = stopwords_removal
        self.__stemming: bool = stemming
        self.__lemmatization: bool = lemmatization
        self.__named_entity_recognition: bool = named_entity_recognition
        self.__strip_multiple_whitespaces: bool = strip_multiple_whitespaces

    @abstractmethod
    def process(self, field_data):
        pass
