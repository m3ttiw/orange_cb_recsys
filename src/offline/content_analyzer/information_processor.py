from abc import ABC, abstractmethod


class InformationProcessor(ABC):
    """
    General class for data processing.
    """
    def __init__(self):
        pass

    @abstractmethod
    def process(self, field_data):
        pass


class ImageProcessor(InformationProcessor):
    """
    Abstract class for image processing.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def process(self, field_data):
        pass


class AudioProcessor(InformationProcessor):
    """
    Abstract class for audio processing.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def process(self, field_data):
        pass


class TextProcessor(InformationProcessor):
    """
    Abstract class for raw text processing.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def process(self, field_data):
        pass


class NLP(TextProcessor):
    """
    Class for processing a text via Natural Language Processing.

    Attributes:
        stopwords_removal (bool): Whether you want to remove stop words.
        stemming (bool): Whether you want to execute stemming.
        lemmatization (bool):  Whether you want to execute lemmatization
        named_entity_recognition (bool): Whether you want to execute named entity recognition
        strip_multiple_whitespaces (bool): Whether you want to remove multiple whitespaces.
    """
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
