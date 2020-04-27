from abc import ABC, abstractmethod


class InformationInterface(ABC):
    """
    Abstract class that deals with the serialization and deserialization of a field (of an item) content
    basing on the type of element extracted.

    Args:
        directory (str): directory where to store the serialized content and where to access for deserialization
    """
    def __init__(self, directory: str):
        self.__directory: str = directory

    @abstractmethod
    def load(self, item_id: str, field_name: str):
        pass

    @abstractmethod
    def serialize(self, field_data):
        """
        Serialize the raw data of a field
        Args:
            field_data: data to serialize
        """
        pass


class ImageInterface(InformationInterface):
    """
    Future feature
    Abstract class to use when the field information is in image format.
    """
    def __init__(self, directory: str):
        super().__init__(directory)

    @abstractmethod
    def load(self, item_id: str, field_name: str):
        pass

    @abstractmethod
    def serialize(self, field_data):
        """
        Abstract method
        """
        pass


class AudioInterface(InformationInterface):
    """
    Future feature
    Abstract class to use when the field information is in audio format.
    """
    def __init__(self, directory: str):
        super().__init__(directory)

    @abstractmethod
    def load(self, item_id: str, field_name: str):
        pass

    @abstractmethod
    def serialize(self, field_data):
        """
        Abstract method
        """
        pass


class TextInterface(InformationInterface):
    """
    Abstract class to use when the field information is textual.
    """
    def __init__(self, directory: str):
        super().__init__(directory)

    @abstractmethod
    def load(self, item_id: str, field_name: str):
        pass

    @abstractmethod
    def serialize(self, field_data):
        """
        Abstract method
        """
        pass
