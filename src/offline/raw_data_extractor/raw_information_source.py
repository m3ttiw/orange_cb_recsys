from abc import ABC, abstractmethod


class RawInformationSource(ABC):
    """
    Abstract Class which deals with generalizing the acquisition of raw descriptions of the fields of the items
    from one of the possible acquisition channels.
    """
    def __init__(self):
        pass

    @abstractmethod
    def extract_field_data(self, item_id: str,
                           field_name: str):
        """
        Abstract method that extract data of a field for an item
        Args:
            item_id (str): item identifier
            field_name (str): name of the field

        Returns:
            data of the given field
        """
        pass


class JSONFile(RawInformationSource):
    """
    Abstract class for the data acquisition from a json file
    """
    def __init__(self, file_path: str):
        """
        Args:
            file_path (str): path of the file
        """
        super().__init__()
        self.__file_path: str = file_path

    def extract_field_data(self, item_id: str,
                           field_name: str):
        print("raw data loading")


class CSVFile(RawInformationSource):
    """
    Abstract class for the data acquisition from a csv file
    """
    def __init__(self, file_path: str):
        """
        Args:
            file_path (str): path of the file
        """
        super().__init__()
        self.__file_path: str = file_path

    def extract_field_data(self, item_id: str,
                           field_name: str):
        pass


class TextFile(RawInformationSource):
    """
    Abstract class for the data acquisition from a text file
    Args:
            file_path (str): path of the file
    """
    def __init__(self, file_path: str):
        super().__init__()
        self.__file_path: str = file_path

    def extract_field_data(self, item_id: str,
                           field_name: str):
        pass


class SQLDatabase(RawInformationSource):
    """
    Abstract class for the data acquisition from a SQL Database
    Args:
            host (str): host ip of the sql server
            username (str): username for the access
            password (str): password for the access
            database_name (str): name of database
            table_name (str): name of the database table where data is stored
    """
    def __init__(self, host: str,
                 username: str,
                 password: str,
                 database_name: str,
                 table_name: str):
        super().__init__()
        self.__host: str = host
        self.__username: str = username
        self.__password: str = password
        self.__database_name: str = database_name
        self.__table_name: str = table_name

    def extract_field_data(self, item_id: str,
                           field_name: str):
        pass
