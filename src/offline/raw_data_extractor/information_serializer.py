from abc import ABC, abstractmethod


class InformationSerializer(ABC):
    """
    Abstract class which deals the serialization of a field (of an item) based on the type of element extracted.

    Args:
        directory (str): directory where to store the serialized content
    """
    def __init__(self, directory: str):
        self.__directory: str = directory

    @abstractmethod
    def serialize(self, field_data):
        """
        Serialize the raw data of a field
        Args:
            field_data: data extracted
        """
        pass


class TextSerializer(InformationSerializer):
    """
    Abstract class that generalizes the serialization of a text type data extracted from a source
    """
    def __init__(self, directory: str):
        super().__init__(directory)

    @abstractmethod
    def serialize(self, field_data):
        """
        Abstract method
        """
        pass


class ImageSerializer(InformationSerializer):
    """
    Future feature
    """
    def __init__(self, directory: str):
        super().__init__(directory)

    @abstractmethod
    def serialize(self, field_data):
        """
        Abstract method
        """
        pass


class SoundSerializer(InformationSerializer):
    """
    Future feature
    """
    def __init__(self, directory: str):
        super().__init__(directory)

    @abstractmethod
    def serialize(self, field_data):
        """
        Abstract method
        """
        pass
