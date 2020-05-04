import json
import mysql.connector
from _csv import reader
from abc import ABC, abstractmethod


class RawInformationSource(ABC):
    """
    Abstract Class which deals with generalizing the acquisition of raw descriptions of the fields of the items
    from one of the possible acquisition channels.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class JSONFile(RawInformationSource):
    """
    Abstract class for the data acquisition from a json file
    """
    def __init__(self, file_path: str):
        """
        """
        super().__init__()
        self.__file_path: str = file_path

    def __iter__(self):
        with open(self.__file_path) as j:
            for line in j:
                line_dict = json.loads(line)
                yield line_dict


class CSVFile(RawInformationSource):
    """
    Abstract class for the data acquisition from a csv file
    """
    def __init__(self, file_path: str):
        """
        """
        super().__init__()
        self.__file_path: str = file_path

    def __iter__(self):
        with open(self.__file_path) as j:
            csv_reader = reader(j)
            for line in csv_reader:
                yield line


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

        conn = mysql.connector.connect(host=self.__host, user=self.__username, password=self.__password)
        cursor = conn.cursor()
        query = """USE """ + self.__database_name + """;"""
        cursor.execute(query)
        conn.commit()
        self.__conn = conn

    def get_host(self):
        return self.__host

    def get_username(self):
        return self.__username

    def get_password(self):
        return self.__password

    def get_database_name(self):
        return self.__database_name

    def get_table_name(self):
        return self.__table_name

    def get_conn(self):
        return self.__conn

    def set_host(self, host):
        self.__host = host

    def set_username(self, username):
        self.__username = username

    def set_password(self, password):
        self.__password = password

    def set_database_name(self, database_name):
        self.__database_name = database_name

    def set_table_name(self, table_name):
        self.__table_name = table_name

    def set_conn(self, conn):
        self.__conn = conn

    def __iter__(self):
        cursor = self.get_conn().cursor(dictionary=True)
        query = """SELECT * FROM """ + self.get_table_name() + """;"""
        cursor.execute(query)
        results = cursor.fetchall()     #array di dizionari
        for result in results:
            yield result
