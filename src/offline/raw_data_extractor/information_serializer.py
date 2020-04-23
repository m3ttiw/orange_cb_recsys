from abc import ABC, abstractmethod


class InformationSerializer(ABC):
    """
    Abstract class which deals
    """
    def __init__(self, directory: str):
        """

        :param directory:
        """
        self.__directory: str = directory

    @abstractmethod
    def serialize(self, field_data):
        """
        abstract method
        :param field_data:
        :return:
        """
        pass


class TextSerializer(InformationSerializer):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def serialize(self, field_data):
        pass


class ImageSerializer(InformationSerializer):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def serialize(self, field_data):
        pass


class SoundSerializer(InformationSerializer):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def serialize(self, field_data):
        pass