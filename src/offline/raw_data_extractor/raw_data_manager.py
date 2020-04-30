from typing import Dict, List

from offline.memory_interfaces.memory_interfaces import InformationInterface
from offline.raw_data_extractor.raw_information_source import RawInformationSource


class RawFieldPipeline: # passaggi per estrarre e serializzare contenuto di un field
    """
    The pipeline for extracting and serializing a field of an item

    Args:
        field_source (RawInformationSource): data source for the associated field
        memory_interface (InformationSerializer): instance to use for serializing the field data
    """
    def __init__(self, field_source: RawInformationSource,
                 memory_interface: InformationInterface):
        self.__field_source: RawInformationSource = field_source
        self.__memory_interface: InformationInterface = memory_interface

    def get_field_source(self):
        return self.__field_source

    def get_memory_interface(self):
        return self.__memory_interface

    def set_field_source(self, field_source: RawInformationSource):
        self.__field_source = field_source

    def set_memory_interface(self, field_serializer: InformationInterface):
        self.__memory_interface = field_serializer


class RawDataConfig:
    """
    Configuration of RawDataManager
    Args:
        fields_pipeline (dict): specifies the source and how to serialize data for the given field.
    """
    def __init__(self, fields_pipeline: Dict[str, RawFieldPipeline] = None):
        if fields_pipeline is None:
            fields_pipeline = {}
        self.__fields_pipeline: Dict[str, RawFieldPipeline] = fields_pipeline

    def add_pipeline(self, field_name: str, field_pipeline: RawFieldPipeline):
        """
        Associate a pipeline process to the field specified by field_name

        Args:
            field_name (str): name of the field
            field_pipeline (RawFieldPipeline): the pipeline for the field

        """
        self.__fields_pipeline[field_name] = field_pipeline

    def get_pipeline(self, field_name: str):
        """
        get the pipeline process of the field identified by field_name

        Args:
            field_name (str): name of the field

        Returns:
            a pipeline process (RawFieldPipeline) of field_name
        """
        return self.__fields_pipeline[field_name]

    def get_field_names(self):
        """
        get the list of field names

        Returns:
            a list of str
        """
        return self.__fields_pipeline.keys()


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

    def start(self):
        """
        Begins to extract data from the source and serializing them according to ways specified in the config
        """
        field_names = self.__config.get_field_names()

        for item_id in self.__item_id_list:
            print("Item " + str(item_id))
            for field_name in field_names:
                print("Field " + field_name)
                field_source = self.__config.get_pipeline(field_name).get_field_source()
                field_data = field_source.extract_field_data(item_id, field_name)
                memory_interface = self.__config.get_pipeline(field_name).get_memory_interface()
                memory_interface.serialize(field_data)
            print("\n")

