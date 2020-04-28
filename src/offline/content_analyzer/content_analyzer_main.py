from offline.memory_interfaces.memory_interfaces import InformationInterface
from src.offline.content_analyzer.field_content_production_technique import FieldContentProductionTechnique
from src.offline.content_analyzer.information_processor import InformationProcessor
from src.offline.content_analyzer.item_representation.item import Item, RepresentedItems
from src.offline.content_analyzer.item_representation.item_field import ItemField

from typing import List, Dict


class FieldContentPipeline:
    """
    The pipeline which specifies the loader, the content_technique and, if necessary, the preprocessor for one
    of the content representations of a field.
    Args:
        memory_interface (InformationLoader):
        content_technique (FieldContentProductionTechnique):
        preprocessor_list (InformationProcessor):
    """
    def __init__(self, memory_interface: InformationInterface,
                 content_technique: FieldContentProductionTechnique,
                 preprocessor_list: List[InformationProcessor] = None):
        if preprocessor_list is None:
            preprocessor_list = []
        self.__memory_interface: InformationInterface = memory_interface
        self.__preprocessor_list: List[InformationProcessor] = preprocessor_list
        self.__content_technique: FieldContentProductionTechnique = content_technique

    def set_memory_interface(self, loader: InformationInterface):
        self.__memory_interface = loader

    def append_preprocessor(self, preprocessor: InformationProcessor):
        self.__preprocessor_list.append(preprocessor)

    def set_content_technique(self, content_technique: FieldContentProductionTechnique):
        self.__content_technique = content_technique

    def get_memory_interface(self):
        return self.__memory_interface

    def get_preprocessor_list(self):
        return self.__preprocessor_list

    def get_content_technique(self):
        return self.__content_technique


class ContentAnalyzerConfig:
    """
    Configuration for the Content analyzer that allows different pipelines to be applied to a specific field, in
    order to represent the field semantic content in different ways.
    Args:
        field_content_pipeline: <field_name, list of pipeline>
    """
    def __init__(self, field_content_pipeline: Dict[str, List[FieldContentPipeline]] = None):
        if field_content_pipeline is None:
            field_content_pipeline = {}
        self.__field_content_pipeline: Dict[str, List[FieldContentPipeline]] = field_content_pipeline

    def add_pipeline(self, field_name: str, pipeline: FieldContentPipeline):
        """
        Add a pipeline for processing a field
        Args:
            field_name (str): name of the field
            pipeline (FieldContentPipeline): pipeline for processing the field
        """
        if field_name in self.__field_content_pipeline.keys():
            self.__field_content_pipeline[field_name].append(pipeline)
        else:
            self.__field_content_pipeline[field_name] = list()
            self.__field_content_pipeline[field_name].append(pipeline)

    def get_pipeline_list(self, field_name: str):
        """
        Get the list of the pipelines for a field
        Args:
            field_name (str): name of the field

        Returns:
            a list of pipelines for a field
        """
        return self.__field_content_pipeline[field_name]

    def get_field_names(self):
        """
        Get the list of the field names
        Returns:
            a list of str
        """
        return self.__field_content_pipeline.keys()


class ContentAnalyzer:
    """
    Class with which the user of the framework interacts, to whom the control of the content analysis phase
    is delegated, providing the appropriate parameters with the possibility of customization on input data
    and technique with which to obtain semantic descriptions from them.

    Args:
        item_id_list (list): list of item id
        config (ContentAnalyzerConfig): configuration for processing the item fields
    """
    def __init__(self, item_id_list: List[str],
                 config: ContentAnalyzerConfig):
        self.__item_id_list: List[str] = item_id_list
        self.__config: ContentAnalyzerConfig = config

    def set_config(self, config: ContentAnalyzerConfig):
        self.__config = config

    def start(self):
        """
        Begins to process the creation of the items

        Returns:
            list of Item objects
        """
        items_producer = ItemsProducer.get_instance()
        items_producer.set_config(self.__config)
        items = RepresentedItems()
        field_name_list = self.__config.get_field_names()
        for item_id in self.__item_id_list:
            print("CREATING ITEM: " + str(item_id))
            items.append(items_producer.create_item(item_id, field_name_list))
            print("################################################")

        return items


class ItemsProducer:
    """
    Singleton class which encapsulates the creation process of the items.
    The creation process is specified in config of ContentAnalyzer and it is supposed to be the same for each
    item.
    """
    __instance = None

    @staticmethod
    def get_instance():
        """
        returns the singleton instance
        Returns:
            ItemProducer object
        """
        """ Static access method. """
        if ItemsProducer.__instance is None:
            ItemsProducer.__instance = ItemsProducer()
        return ItemsProducer.__instance

    def __init__(self):
        self.__config: ContentAnalyzerConfig = None
        """ Virtually private constructor. """
        if ItemsProducer.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ItemsProducer.__instance = self

    def set_config(self, config: ContentAnalyzerConfig):
        """
        Set the config of ContentAnalyzer which specifies how to process a item

        Args:
            config (ContentAnalyzerConfig): configuration of ContentAnalyzer
        """
        self.__config = config

    def create_item(self, item_id: str, field_name_list: List[str]):
        """
        Create an item processing every field in the specified way

        Args:
            item_id (str): id of the item
            field_name_list (str): name list of fields in the item

        Returns:
            Item object

        Raises:
            general Exception
        """
        if self.__config is None:
            raise Exception("You must set a config with set_config()")
        else:
            item = Item(item_id)
            for field_name in field_name_list:
                print("Creating field:", field_name)
                pipeline_list = self.__config.get_pipeline_list(field_name)
                field = ItemField(field_name)
                i = 1
                for pipeline in pipeline_list:
                    print("Representation", str(i), " for field", field_name)
                    memory_interface = pipeline.get_memory_interface()
                    field_data = memory_interface.load(item_id, field_name)
                    preprocessor_list = pipeline.get_preprocessor_list()
                    for preprocessor in preprocessor_list:
                        field_data = preprocessor.process(field_data)

                    content_technique = pipeline.get_content_technique()
                    field.append(content_technique.produce_content(field_data))
                    i += 1
                    print("---------------------------------")
                item.append(field)
                print("\n")

            return item
