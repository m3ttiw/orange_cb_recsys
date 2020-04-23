from abc import ABC, abstractmethod


class InformationSerializer(ABC):
    def __init__(self, directory: str):
        self.__directory: str = directory

    @abstractmethod
    def serialize(self, field_data):
        pass


class TextSerializer():
    def __init__(self):
        super().__init__()

    @abstractmethod
    def serialize(self, field_data):
        pass


class ImageSerializer():
    def __init__(self):
        super().__init__()

    @abstractmethod
    def serialize(self, field_data):
        pass


class SoundSerializer():
    def __init__(self):
        super().__init__()

    @abstractmethod
    def serialize(self, field_data):
        pass