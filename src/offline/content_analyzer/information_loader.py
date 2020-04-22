from abc import ABC, abstractmethod


class InformationLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load(self, item_id: str, field_name: str):
        pass


class ImageLoader(InformationLoader):
    """
    Future Feature
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, item_id: str, field_name: str):
        pass


class AudioLoader(InformationLoader):
    """
    Future Feature
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, item_id: str, field_name: str):
        pass


class TextLoader(InformationLoader):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, item_id: str, field_name: str):
        pass
