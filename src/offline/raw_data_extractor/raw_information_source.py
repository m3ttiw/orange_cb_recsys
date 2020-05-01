from abc import ABC, abstractmethod


class RawInformationSource(ABC):
    """
    Abstract Class which deals with generalizing the acquisition of raw descriptions of the fields of the items
    from one of the possible acquisition channels.
    """
    def __init__(self, id_field_name: str):
        self.__id_field_name = id_field_name

    @abstractmethod
    def extract_field_data(self, field_name: str, position: int):
        """
        Abstract method that extract data of a field for an item
        Args:
            position:
            field_name (str): name of the field

        Returns:
            data of the given field
        """
        pass


class JSONFile(RawInformationSource):
    """
    Abstract class for the data acquisition from a json file
    """
    def __init__(self, id_field_name: str, file_path: str):
        """
        """
        super().__init__(id_field_name)
        self.__file_path: str = file_path

    def extract_field_data(self, field_name: str, position: int):
        #estrai il field di nome field_name in base a position
        #position in Json è il numero della riga



        print("raw data loading")


class CSVFile(RawInformationSource):
    """
    Abstract class for the data acquisition from a csv file
    """
    def __init__(self, id_field_name: str, file_path: str):
        """
        """
        super().__init__(id_field_name)
        self.__file_path: str = file_path

    def extract_field_data(self, field_name: str, position: int):
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
    def __init__(self, id_field_name: str,
                 host: str,
                 username: str,
                 password: str,
                 database_name: str,
                 table_name: str):
        super().__init__(id_field_name)
        self.__host: str = host
        self.__username: str = username
        self.__password: str = password
        self.__database_name: str = database_name
        self.__table_name: str = table_name

    def extract_field_data(self, field_name: str, position: int):
        pass
