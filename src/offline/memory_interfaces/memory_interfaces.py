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
    def new_item(self):
        pass

    @abstractmethod
    def new_field(self, field_name: str, field_data):
        """
        Serialize the raw data of a field
        Args:
            field_data: data to serialize
            :param field_name:
        """
        pass

    @abstractmethod
    def serialize_item(self):
        pass

    @abstractmethod
    def init_writing(self):
        pass

    @abstractmethod
    def stop_writing(self):
        pass

    def get_directory(self):
        return self.__directory


class ImageInterface(InformationInterface):
    """
    Future feature
    Abstract class to use when the field information is in image format.
    """
    def __init__(self, directory: str):
        super().__init__(directory)

    @abstractmethod
    def new_item(self):
        pass

    @abstractmethod
    def new_field(self, field_name: str, field_data):
        """
        Abstract method
        """
        pass

    @abstractmethod
    def serialize_item(self):
        pass

    @abstractmethod
    def init_writing(self):
        pass

    @abstractmethod
    def stop_writing(self):
        pass


class AudioInterface(InformationInterface):
    """
    Future feature
    Abstract class to use when the field information is in audio format.
    """
    def __init__(self, directory: str):
        super().__init__(directory)

    @abstractmethod
    def new_item(self):
        pass

    @abstractmethod
    def new_field(self, field_name: str, field_data):
        """
        Abstract method
        """
        pass

    @abstractmethod
    def serialize_item(self):
        pass

    @abstractmethod
    def init_writing(self):
        pass

    @abstractmethod
    def stop_writing(self):
        pass


class TextInterface(InformationInterface):
    """
    Abstract class to use when the field information is textual.
    """
    def __init__(self, directory: str):
        super().__init__(directory)

    @abstractmethod
    def new_item(self):
        pass

    @abstractmethod
    def new_field(self, field_name: str, field_data):
        """
        Abstract method
        """
        pass

    @abstractmethod
    def serialize_item(self):
        pass

    @abstractmethod
    def init_writing(self):
        pass

    @abstractmethod
    def stop_writing(self):
        pass
