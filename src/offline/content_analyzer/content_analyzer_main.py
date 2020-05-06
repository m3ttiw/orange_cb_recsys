from typing import List, Dict

from src.offline.content_analyzer.content_representation.content import RepresentedContents, Content
from src.offline.content_analyzer.content_representation.content_field import ContentField
from src.offline.content_analyzer.field_content_production_technique \
    import FieldContentProductionTechnique
from src.offline.content_analyzer.information_processor import InformationProcessor
from src.offline.raw_data_extractor.raw_information_source import RawInformationSource


class FieldRepresentationPipeline:
    """
    Pipeline which specifies:
     a list of pre-processing techniques(optional) and a content production techniques
    to specify how to produce one of the representations of a field.
    Args:
        representation_name (str): name that will be assigned to the produced representation
        content_technique (FieldContentProductionTechnique):
            used to produce complex representation of the field given pre-processed information
        preprocessor_list (InformationProcessor):
            list of information processors that will be applied on original text,
            in a pipeline way
    """
    def __init__(self, representation_name: str,
                 content_technique: FieldContentProductionTechnique,
                 preprocessor_list: List[InformationProcessor] = None, ):
        if preprocessor_list is None:
            preprocessor_list = []
        self.__representation_name = representation_name
        self.__preprocessor_list: List[InformationProcessor] = preprocessor_list
        self.__content_technique: FieldContentProductionTechnique = content_technique

    def append_preprocessor(self, preprocessor: InformationProcessor):
        self.__preprocessor_list.append(preprocessor)

    def set_content_technique(self, content_technique: FieldContentProductionTechnique):
        self.__content_technique = content_technique

    def get_preprocessor_list(self) -> List[InformationProcessor]:
        return self.__preprocessor_list

    def get_content_technique(self) -> FieldContentProductionTechnique:
        return self.__content_technique

    def get_representation_name(self) -> str:
        return self.__representation_name


class FieldConfig:
    """
    Class that represent the config for a field.
    Args:
        pipelines_list (List<FieldRepresentationPipeline>):
            list of pipeline that will be used to produce different field's representations,
            one pipeline for each representation
    """
    def __init__(self, pipelines_list: List[FieldRepresentationPipeline] = None):
        if pipelines_list is None:
            pipelines_list = []

        self.__pipelines_list: List[FieldRepresentationPipeline] = pipelines_list

    def append_pipeline(self, pipeline: FieldRepresentationPipeline):
        self.__pipelines_list.append(pipeline)

    def get_pipeline_list(self) -> List[FieldRepresentationPipeline]:
        return self.__pipelines_list


class ContentAnalyzerConfig:
    """
    Class that represent the Configuration for the content analyzer,
    Args:
        source (RawInformationSource):
            raw data source to iterate on for item's extraction
        id_field_name (str): name of the field where the item_id can be found
        field_config_dict (Dict<str, FieldConfig>):
            store the config for each field_name
    """
    def __init__(self, source: RawInformationSource,
                 id_field_name: str,
                 field_config_dict: Dict[str, FieldConfig] = None):
        if field_config_dict is None:
            field_config_dict = {}
        self.__field_config_dict: Dict[str, FieldConfig] = field_config_dict
        self.__source: RawInformationSource = source
        self.__id_field_name: str = id_field_name

    def get_id_field_name(self) -> str:
        return self.__id_field_name

    def get_source(self) -> RawInformationSource:
        return self.__source

    def get_pipeline_list(self, field_name: str) -> List[FieldRepresentationPipeline]:
        """
        Get the list of the pipelines specified for the input field
        Args:
            field_name (str): name of the field

        Returns:
            List<FieldRepresentationPipeline>:
                the list of pipelines specified for the input field
        """
        return self.__field_config_dict[field_name].get_pipeline_list()

    def get_field_names(self) -> List[str]:
        """
        Get the list of the field names
        Returns:
            List<str>: list of config dictionary keys
        """
        return self.__field_config_dict.keys()

    def append_field_config(self, field_name: str, field_config: FieldConfig):
        self.__field_config_dict[field_name] = field_config


class ContentAnalyzer:
    """
    Class to whom the control of the content analysis phase is delegated,
    providing the appropriate parameters in the config,
    config objects provide possibility of customization on input data and how to process them.

    Args:
        config (ContentAnalyzerConfig): configuration for processing the item fields
    """
    def __init__(self, config: ContentAnalyzerConfig):
        self.__config: ContentAnalyzerConfig = config

    def set_config(self, config: ContentAnalyzerConfig):
        self.__config = config

    def fit(self) -> RepresentedContents:
        """
        Begins to process the creation of the contents

        Returns:
            List<Content>:
                list which elements are the produced content instances
        """
        contents_producer = ContentsProducer.get_instance()
        contents_producer.set_config(self.__config)
        contents = RepresentedContents()
        for raw_content in self.__config.get_source():
            contents.append(contents_producer.create_content(raw_content))

        return contents


class ContentsProducer:
    """
    Singleton class which encapsulates the creation process of the items,
    The creation process is specified in config of ContentAnalyzer and
    it is supposed to be the same for each item.
    """
    __instance = None

    @staticmethod
    def get_instance():
        """
        returns the singleton instance
        Returns:
            ItemProducer: instance
        """
        # Static access method
        if ContentsProducer.__instance is None:
            ContentsProducer.__instance = ContentsProducer()
        return ContentsProducer.__instance

    def __init__(self):
        self.__config: ContentAnalyzerConfig = None
        # Virtually private constructor.
        if ContentsProducer.__instance is not None:
            raise Exception("This class is a singleton!")
        ContentsProducer.__instance = self

    def set_config(self, config: ContentAnalyzerConfig):
        self.__config = config

    def create_content(self, raw_content: Dict):
        """
        Create an item processing every field in the specified way,
        this class be iteratively invoked by the fit method

        Returns:
            Content: an instance of content with his fields

        Raises:
            general Exception
        """
        if self.__config is None:
            raise Exception("You must set a config with set_config()")

        content = Content(raw_content[self.__config.get_id_field_name()])
        field_name_list = self.__config.get_field_names()
        for field_name in field_name_list:
            print("Creating field:", field_name)
            pipeline_list = self.__config.get_pipeline_list(field_name)
            field = ContentField(field_name)
            for i, pipeline in enumerate(pipeline_list):
                print("Representation", str(i), " for field", field_name)
                field_data = raw_content[field_name]
                preprocessor_list = pipeline.get_preprocessor_list()
                for preprocessor in preprocessor_list:
                    field_data = preprocessor.process(field_data)

                content_technique = pipeline.get_content_technique()
                field.append(content_technique.produce_content(field_data))
                print("---------------------------------")
            content.append(field)
            print("\n")

        return content
