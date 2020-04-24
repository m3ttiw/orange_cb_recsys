from abc import ABC, abstractmethod


class InformationLoader(ABC):
    """
    Abstract Class for loading raw information about an item's field.
    """
    def __init__(self):
        pass

    @abstractmethod
    def load(self, item_id: str, field_name: str):
        pass


class ImageLoader(InformationLoader):
    """
    Abstract class to use when the field information is in image format.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, item_id: str, field_name: str):
        pass


class AudioLoader(InformationLoader):
    """
    Abstract class to use when the field information is in audio format.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, item_id: str, field_name: str):
        pass


class TextLoader(InformationLoader):
    """
    Abstract class to use when the field information is textual.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, item_id: str, field_name: str):
        pass
