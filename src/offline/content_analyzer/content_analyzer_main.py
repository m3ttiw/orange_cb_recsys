from typing import List, Dict, Tuple

from src.offline.content_analyzer.content_representation.content import RepresentedContents, Content
from src.offline.content_analyzer.content_representation.content_field import ContentField
from src.offline.content_analyzer.field_content_production_technique import FieldContentProductionTechnique
from src.offline.content_analyzer.information_processor import InformationProcessor
from src.offline.raw_data_extractor.raw_information_source import RawInformationSource


class FieldRepresentationPipeline:
    """
    The pipeline which specifies the loader, the content_technique and, if necessary, the preprocessor for one
    of the content representations of a field.
    Args:
        content_technique (FieldContentProductionTechnique):
        preprocessor_list (InformationProcessor):
    """

    def __init__(self, content_technique: FieldContentProductionTechnique,
                 preprocessor_list: List[InformationProcessor] = None):
        if preprocessor_list is None:
            preprocessor_list = []
        self.__preprocessor_list: List[InformationProcessor] = preprocessor_list
        self.__content_technique: FieldContentProductionTechnique = content_technique

    def append_preprocessor(self, preprocessor: InformationProcessor):
        self.__preprocessor_list.append(preprocessor)

    def set_content_technique(self, content_technique: FieldContentProductionTechnique):
        self.__content_technique = content_technique

    def get_preprocessor_list(self):
        return self.__preprocessor_list

    def get_content_technique(self):
        return self.__content_technique


class FieldConfig:
    def __init__(self, pipeline_list: List[FieldRepresentationPipeline] = None):
        if pipeline_list is None:
            pipeline_list = []

        self.__pipeline_list: List[FieldRepresentationPipeline] = pipeline_list

    def add_pipeline(self, pipeline: FieldRepresentationPipeline):
        """
        Add a pipeline for processing a field
        Args:
            pipeline (FieldRepresentationPipeline): pipeline for processing the field
        """

        self.__pipeline_list.append(pipeline)

    def get_pipeline_list(self):
        return self.__pipeline_list


class ContentAnalyzerConfig:
    """
    Configuration for the Content analyzer that allows different pipelines to be applied to a specific field, in
    order to represent the field semantic content in different ways.
    Args:
        fields_config: <field_name, list of pipeline>
    """

    def __init__(self, source: RawInformationSource,
                 id_field_name: str,
                 fields_config: Dict[str, FieldConfig] = None):
        if fields_config is None:
            fields_config = {}
        self.__fields_config: Dict[str, FieldConfig] = fields_config
        self.__source: RawInformationSource = source
        self.__id_field_name: str = id_field_name

    def get_id_field_name(self):
        return self.__id_field_name

    def get_source(self):
        return self.__source

    def get_pipeline_list(self, field_name: str):
        """
        Get the list of the pipelines for a field
        Args:
            field_name (str): name of the field

        Returns:
            a list of pipelines for a field
        """
        return self.__fields_config[field_name].get_pipeline_list()

    def get_field_names(self):
        """
        Get the list of the field names
        Returns:
            a list of str
        """
        return self.__fields_config.keys()

    def append_field_config(self, field_name: str, field_config: FieldConfig):
        self.__fields_config[field_name] = field_config


class ContentAnalyzer:
    """
    Class with which the user of the framework interacts, to whom the control of the content analysis phase
    is delegated, providing the appropriate parameters with the possibility of customization on input data
    and technique with which to obtain semantic descriptions from them.

    Args:
        config (ContentAnalyzerConfig): configuration for processing the item fields
    """

    def __init__(self, config: ContentAnalyzerConfig):
        self.__config: ContentAnalyzerConfig = config

    def set_config(self, config: ContentAnalyzerConfig):
        self.__config = config

    def fit(self):
        """
        Begins to process the creation of the items

        Returns:
            list of Item objects
        """
        contents_producer = ContentsProducer.get_instance()
        contents_producer.set_config(self.__config)
        contents = RepresentedContents()
        for raw_content in self.__config.get_source():
            contents.append(contents_producer.create_content(raw_content))

        return contents


class ContentsProducer:
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
        if ContentsProducer.__instance is None:
            ContentsProducer.__instance = ContentsProducer()
        return ContentsProducer.__instance

    def __init__(self):
        self.__config: ContentAnalyzerConfig = None
        """ Virtually private constructor. """
        if ContentsProducer.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ContentsProducer.__instance = self

    def set_config(self, config: ContentAnalyzerConfig):
        """
        Set the config of ContentAnalyzer which specifies how to process a item

        Args:
            config (ContentAnalyzerConfig): configuration of ContentAnalyzer
        """
        self.__config = config

    def create_content(self, raw_content: Dict):
        """
        Create an item processing every field in the specified way

        Returns:
            Item object

        Raises:
            general Exception
        """
        if self.__config is None:
            raise Exception("You must set a config with set_config()")
        else:
            content = Content(raw_content[self.__config.get_id_field_name()])
            field_name_list = self.__config.get_field_names()
            for field_name in field_name_list:
                print("Creating field:", field_name)
                pipeline_list = self.__config.get_pipeline_list(field_name)
                field = ContentField(field_name)
                i = 1
                for pipeline in pipeline_list:
                    print("Representation", str(i), " for field", field_name)
                    field_data = raw_content[field_name]
                    preprocessor_list = pipeline.get_preprocessor_list()
                    for preprocessor in preprocessor_list:
                        field_data = preprocessor.process(field_data)

                    content_technique = pipeline.get_content_technique()
                    field.append(content_technique.produce_content(str(i), field_data,
                                                                   field_name=field_name,
                                                                   item_id=raw_content[self.__config.get_id_field_name()]))
                    i += 1
                    print("---------------------------------")
                content.append(field)
                print("\n")

            return content
