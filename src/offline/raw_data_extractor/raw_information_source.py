from abc import ABC, abstractmethod


class RawInformationSource(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extract_field_data(self, item_id: str,
                           field_name: str):
        pass


class CSVFile(RawInformationSource):
    def __init__(self, file_path: str):
        super().__init__()
        self.__file_path: str = file_path

    @abstractmethod
    def extract_field_data(self, item_id: str,
                           field_name: str):
        pass


class TextFile(RawInformationSource):
    def __init__(self, file_path: str):
        super().__init__()
        self.__file_path: str = file_path

    @abstractmethod
    def extract_field_data(self, item_id: str,
                           field_name: str):
        pass


class SQLDatabase(RawInformationSource):
    def __init__(self, host: str, username: str, password: str, database_name: str, table_name: str):
        super().__init__()
        self.__host: str = host
        self.__username: str = username
        self.__password: str = password
        self.__database_name: str = database_name
        self.__table_name: str = table_name

    @abstractmethod
    def extract_field_data(self, item_id: str,
                           field_name: str):
        pass
