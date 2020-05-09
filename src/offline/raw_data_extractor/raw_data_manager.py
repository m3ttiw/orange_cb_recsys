from typing import Dict

from src.offline.memory_interfaces.memory_interfaces import InformationInterface
from src.offline.raw_data_extractor.raw_information_source import RawInformationSource
from src.offline.utils.id_merger import id_merger


class RawDataConfig:
    """
    Configuration of RawDataManager.
    Args:
        source (RawInformationSource): raw data source from which extract the content
        id_field_name (str): name of the field that represents the item id
        fields_interface (InformationInterface):
            specifies for each field
            which interface use to serialize field data
    """
    def __init__(self, source: RawInformationSource = None,
                 id_field_name: str = None,
                 fields_interface: Dict[str, InformationInterface] = None):
        if fields_interface is None:
            fields_interface = {}
        self.__fields_interface: Dict[str, InformationInterface] = fields_interface
        self.__id_field_name: str = id_field_name
        self.__source: RawInformationSource = source

    def set_source(self, source: str):
        self.__source = source

    def get_source(self) -> RawInformationSource:
        return self.__source

    def set_id_field_name(self, id_field_name: str):
        self.__id_field_name = id_field_name

    def get_id_field_name(self) -> str:
        return self.__id_field_name

    def set_interface(self, field_name: str, field_interface: InformationInterface):
        self.__fields_interface[field_name] = field_interface

    def get_interface(self, field_name: str) -> InformationInterface:
        return self.__fields_interface[field_name]

    def get_field_names(self):
        """
        get the list of field names

        Returns:
            List<str>: list of config dict keys
        """
        return self.__fields_interface.keys()

    def get_interfaces(self):
        """
        get the list of field interfaces

        Returns:
            List<InformationInterface>: list of config dict values
        """
        return set(self.__fields_interface.values())


class RawDataManager:
    """
    Class to carry out the steps of this phase,
    then data extraction and data serialization according to the config.

    Args:
        config (RawDataConfig): manager configuration
    """
    def __init__(self, config: RawDataConfig):
        self.__config: RawDataConfig = config

    def fit(self):
        """
        Begins to extract data from the source
        and serializing them according to ways specified in the config
        """
        print("####################### FASE 1 #########################")
        CONTENT_ID = "content_id"
        field_names = self.__config.get_field_names()
        interfaces = self.__config.get_interfaces()
        for interface in interfaces:
            interface.init_writing()
        i = 0
        for content in self.__config.get_source():
            print("Content:", i)
            id_values = []
            for id_field_name in self.__config.get_id_field_name():
                id_values.append(content[id_field_name])
            content_id = id_merger(id_values)
            for interface in interfaces:
                interface.new_content()
                interface.new_field(CONTENT_ID, content_id)
            for field_name in field_names:
                print("Field " + field_name)
                field_data = content[field_name]
                memory_interface = self.__config.get_interface(field_name)
                memory_interface.new_field(field_name, field_data)
            for interface in interfaces:
                interface.serialize_content()
            i += 1
            print("\n")

        for interface in interfaces:
            interface.stop_writing()
