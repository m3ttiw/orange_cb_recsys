import json
from typing import Dict, List

from offline.memory_interfaces.memory_interfaces import InformationInterface
from offline.memory_interfaces.text_interface import IndexInterface
from offline.raw_data_extractor.raw_information_source import RawInformationSource


class RawDataConfig:
    """
    Configuration of RawDataManager
    Args:
        fields_pipeline (dict): specifies the source and how to serialize data for the given field.
    """
    def __init__(self, json_path: str = None,
                 id_field_name: str = None,
                 fields_interface: Dict[str, InformationInterface] = None):
        # dobbiamo specificare come serializzare
        if fields_interface is None:
            fields_interface = {}
        self.__fields_interface: Dict[str, InformationInterface] = fields_interface
        self.__id_field_name = id_field_name
        self.__json_path = json_path

    def set_json_path(self, json_path: str):
        self.__json_path = json_path

    def get_json_path(self):
        return self.__json_path

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


class RawDataManager:
    """
    Class with which the user of the framework interacts to carry out the steps of this phase,
    then data extraction and data serialization.

    Args:
        item_id_list (list): list of items id
        config (RawDataConfig): manager configuration
    """
    def __init__(self, item_id_list: List[str],
                 config: RawDataConfig):
        self.__item_id_list: List[str] = item_id_list
        self.__config: RawDataConfig = config

    def fit(self):
        """
        Begins to extract data from the source and serializing them according to ways specified in the config
        """
        field_names = self.__config.get_field_names()

        id_interface = self.__config.get_interface(self.__config.get_id_field_name())
        with open(self.__config.get_json_path()) as j:
            for line in j:
                data = json.loads(line)
                item_id = data[self.__config.get_id_field_name()]
                id_interface.serialize(item_id)
                for field_name in field_names:
                    print("Field " + field_name)
                    field_source = self.__config.get_pipeline(field_name).get_field_source()
                    field_data = field_source.extract_field_data(field_name)
                    memory_interface = self.__config.get_pipeline(field_name).get_memory_interface()
                    memory_interface.serialize(field_data, item_id)
                print("\n")


        for item_id in self.__item_id_list:
            #estraiamo sempre l'id
            print("Item " + str(item_id))
            for field_name in field_names:
                print("Field " + field_name)
                field_source = self.__config.get_pipeline(field_name).get_field_source()
                field_data = field_source.extract_field_data(field_name)
                memory_interface = self.__config.get_pipeline(field_name).get_memory_interface()
                memory_interface.serialize(field_data)
            print("\n")



