from abc import ABC, abstractmethod


class InformationInterface(ABC):
    """
    Abstract class that deals with the serialization
    and deserialization of a field (of a content) data
    basing on the type of element extracted.

    Args:
        directory (str):
            directory where to store the serialized content and where to access for deserialization
    """
    def __init__(self, directory: str):
        self.__directory: str = directory

    @abstractmethod
    def new_content(self):
        """
        Creates a new item, that will be serialized by the apposite method.
        """

    @abstractmethod
    def new_field(self, field_name: str, field_data):
        """
        Serialize the raw data of a field.
        Args:
            field_data: data to serialize
            field_name: name of the created field
        """

    @abstractmethod
    def serialize_item(self):
        """
        Add to the serialized collection the current item
        """

    @abstractmethod
    def init_writing(self):
        """
        Set the interface in writing mode,
        if the specified directory oes not exit a new one will be created
        """

    @abstractmethod
    def stop_writing(self):
        """
        Stop writing mode
        """

    def get_directory(self):
        return self.__directory


class ImageInterface(InformationInterface):
    """
    Future feature
    Abstract class to use when the field information is in image format.
    """
    @abstractmethod
    def new_content(self):
        pass

    @abstractmethod
    def new_field(self, field_name: str, field_data):
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
    @abstractmethod
    def new_content(self):
        pass

    @abstractmethod
    def new_field(self, field_name: str, field_data):
        """
        Abstract method
        """

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
    @abstractmethod
    def new_content(self):
        pass

    @abstractmethod
    def new_field(self, field_name: str, field_data):
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
