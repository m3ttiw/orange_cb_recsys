import json
from typing import Dict, List

from offline.memory_interfaces.memory_interfaces import InformationInterface
from offline.memory_interfaces.text_interface import IndexInterface
from offline.raw_data_extractor.raw_information_source import RawInformationSource


class RawDataConfig:
    """
    Configuration of RawDataManager
    Args:

    """
    def __init__(self, source: RawInformationSource = None,
                 id_field_name: str = None,
                 fields_interface: Dict[str, InformationInterface] = None):
        if fields_interface is None:
            fields_interface = {}
        self.__fields_interface: Dict[str, InformationInterface] = fields_interface
        self.__id_field_name = id_field_name
        self.__source = source

    def set_source(self, source: str):
        self.__source = source

    def get_source(self):
        return self.__source

    def set_id_field_name(self, id_field_name: str):
        self.__id_field_name = id_field_name

    def get_id_field_name(self):
        return self.__id_field_name

    def add_interface(self, field_name: str, field_interface: InformationInterface):
        """
        Associate a pipeline process to the field specified by field_name

        Args:
            field_interface:
            field_name (str): name of the field

        """
        self.__fields_interface[field_name] = field_interface

    def get_interface(self, field_name: str):
        """
        get the pipeline process of the field identified by field_name

        Args:
            field_name (str): name of the field

        Returns:
            a pipeline process (RawFieldPipeline) of field_name
        """
        return self.__fields_interface[field_name]

    def get_field_names(self):
        """
        get the list of field names

        Returns:
            a list of str
        """
        return self.__fields_interface.keys()

    def get_interfaces(self):
        """
        get the list of field interfaces

        Returns:
            a list of str
        """
        return set(self.__fields_interface.values())


class RawDataManager:
    """
    Class with which the user of the framework interacts to carry out the steps of this phase,
    then data extraction and data serialization.

    Args:
        config (RawDataConfig): manager configuration
    """
    def __init__(self, config: RawDataConfig):
        self.__config: RawDataConfig = config

    def fit(self):
        """
        Begins to extract data from the source and serializing them according to ways specified in the config
        """

        CONTENT_ID = "content_id"
        field_names = self.__config.get_field_names()
        interfaces = self.__config.get_interfaces()
        for item in self.__config.get_source():
            for interface in interfaces:
                interface.new_item()
                interface.serialize(CONTENT_ID, item[self.__config.get_id_field_name()])

            for field_name in field_names:
                print("Field " + field_name)
                field_data = item[field_name]
                memory_interface = self.__config.get_interface(field_name)
                memory_interface.serialize(field_name, field_data)
            for interface in interfaces:
                interface.close_item()
            print("\n")
