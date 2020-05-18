from typing import List, Dict, Tuple, Set
import time
import os

from offline.memory_interfaces.memory_interfaces import InformationInterface
from src.offline.utils.id_merger import id_merger
from src.offline.content_analyzer.content_representation.content import RepresentedContents, Content
from src.offline.content_analyzer.content_representation.content_field import ContentField
from src.offline.content_analyzer.field_content_production_technique \
    import FieldContentProductionTechnique, CollectionBasedTechnique, SingleContentTechnique
from src.offline.content_analyzer.information_processor import InformationProcessor
from src.offline.raw_data_extractor.raw_information_source import RawInformationSource

parent_dir = '../../contents'


class FieldRepresentationPipeline:
    """
    Pipeline which specifies:
     a list of pre-processing techniques(optional) and a content production technique
    to specify how to produce one of the representations of a field.
    Args:
        content_technique (FieldContentProductionTechnique):
            used to produce complex representation of the field given pre-processed information
        preprocessor_list (InformationProcessor):
            list of information processors that will be applied on original text,
            in a pipeline way
    """

    instance_counter: int = 0

    def __init__(self, content_technique: FieldContentProductionTechnique,
                 preprocessor_list: List[InformationProcessor] = None):
        if preprocessor_list is None:
            preprocessor_list = []
        self.__preprocessor_list: List[InformationProcessor] = preprocessor_list
        self.__content_technique: FieldContentProductionTechnique = content_technique
        self.__id: str = str(FieldRepresentationPipeline.instance_counter)
        FieldRepresentationPipeline.instance_counter += 1

    def append_preprocessor(self, preprocessor: InformationProcessor):
        self.__preprocessor_list.append(preprocessor)

    def set_content_technique(self, content_technique: FieldContentProductionTechnique):
        self.__content_technique = content_technique

    def get_preprocessor_list(self) -> List[InformationProcessor]:
        return self.__preprocessor_list

    def get_content_technique(self) -> FieldContentProductionTechnique:
        return self.__content_technique

    def __str__(self):
        return self.__id

    def __repr__(self):
        msg = "< " + "FieldRepresentationPipeline: " + "" \
            "preprocessor_list = " + str(self.__preprocessor_list) + "; " \
            "content_technique = " + str(self.__content_technique) + ">"
        return msg


class FieldConfig:
    """
    Class that represent the config for a field.
    Args:
        pipelines_list (List<FieldRepresentationPipeline>):
            list of pipeline that will be used to produce different field's representations,
            one pipeline for each representation
    """

    def __init__(self, memory_interface: InformationInterface = None,
                 pipelines_list: List[FieldRepresentationPipeline] = None):
        if pipelines_list is None:
            pipelines_list = []
        self.__memory_interface: InformationInterface = memory_interface
        self.__pipelines_list: List[FieldRepresentationPipeline] = pipelines_list

    def get_memory_interface(self) -> InformationInterface:
        return self.__memory_interface

    def set_memory_interface(self, memory_interface: InformationInterface):
        self.__memory_interface = memory_interface

    def append_pipeline(self, pipeline: FieldRepresentationPipeline):
        self.__pipelines_list.append(pipeline)

    def get_pipeline_list(self) -> List[FieldRepresentationPipeline]:
        return self.__pipelines_list

    def __str__(self):
        return "FieldConfig"

    def __repr__(self):
        return "< " + "FieldConfig: " + "" \
                "pipelines_list = " + str(self.__pipelines_list) + " >"


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

    def __init__(self, content_type: str,
                 source: RawInformationSource,
                 id_field_name,
                 output_directory: str,
                 field_config_dict: Dict[str, FieldConfig] = None):
        if field_config_dict is None:
            field_config_dict = {}
        self.__output_directory: str = output_directory + '_' + str(time.time())
        self.__content_type = content_type
        self.__field_config_dict: Dict[str, FieldConfig] = field_config_dict
        self.__source: RawInformationSource = source
        self.__id_field_name: str = id_field_name

    def get_output_directory(self):
        return self.__output_directory

    def get_content_type(self):
        return self.__content_type

    def get_id_field_name(self):
        return self.__id_field_name

    def get_source(self) -> RawInformationSource:
        return self.__source

    def get_memory_interface(self, field_name: str) -> InformationInterface:
        return self.__field_config_dict[field_name].get_memory_interface()

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

    def get_field_name_list(self) -> List[str]:
        """
        Get the list of the field names
        Returns:
            List<str>: list of config dictionary keys
        """
        return self.__field_config_dict.keys()

    def get_interfaces(self) -> Set[InformationInterface]:
        """
        get the list of field interfaces

        Returns:
            List<InformationInterface>: list of config dict values
        """
        interfaces = set()
        for key in self.__field_config_dict.keys():
            if self.__field_config_dict[key].get_memory_interface() is not None:
                interfaces.add(self.__field_config_dict[key].get_memory_interface())
        return interfaces

    def append_field_config(self, field_name: str, field_config: FieldConfig):
        self.__field_config_dict[field_name] = field_config

    def get_collection_based_techniques(self) -> Set[CollectionBasedTechnique]:
        techniques = set()
        for field_config in self.__field_config_dict.values():
            for pipeline in field_config.get_pipeline_list():
                if isinstance(pipeline.get_content_technique(), CollectionBasedTechnique):
                    techniques.add(pipeline.get_content_technique())

        return techniques

    def __str__(self):
        return str(self.__id_field_name)

    def __repr__(self):
        msg = "< " + "ContentAnalyzerConfig: " + "" \
                                                 "id_field_name = " + str(self.__id_field_name) + "; " \
                                                                                                  "source = " + str(
            self.__source) + "; " \
                             "field_config_dict = " + str(self.__field_config_dict) + "; " \
                                                                                      "content_type = " + str(
            self.__content_type) + ">"
        return msg


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

    def fit(self):
        """
        Begins to process the creation of the contents

        Returns:
            List<Content>:
                list which elements are the produced content instances
        """
        path = os.path.join(parent_dir, self.__config.get_output_directory())
        os.mkdir(path)

        contents_producer = ContentsProducer.get_instance()
        contents_producer.set_config(self.__config)

        interfaces = self.__config.get_interfaces()
        for interface in interfaces:
            interface.init_writing()

        for field_name in self.__config.get_field_name_list():
            for pipeline in self.__config.get_pipeline_list(field_name):
                if isinstance(pipeline.get_content_technique(), CollectionBasedTechnique):
                    pipeline.get_content_technique(). \
                        append_field_need_refactor(field_name, str(pipeline), pipeline.get_preprocessor_list())

        for technique in self.__config.get_collection_based_techniques():
            technique.dataset_refactor(self.__config.get_source(), self.__config.get_id_field_name())

        i = 0
        for raw_content in self.__config.get_source():
            content = contents_producer.create_content(raw_content)
            print(content)
            content.serialize(path)
            i += 1

    def __str__(self):
        return "ContentAnalyzer"

    def __repr__(self):
        msg = "< " + "ContentAnalyzer: " + "" \
                                           "config = " + str(self.__config) + "; >"
        return msg


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
        else:
            CONTENT_ID = "content_id"
            # search for timestamp as dataset field, no timestamp needed for items
            timestamp = None
            if self.__config.get_content_type() != "ITEM":
                if "timestamp" in raw_content.keys():
                    timestamp = raw_content["timestamp"]
                else:
                    timestamp = time.time()

            # construct id from list of fields that compound id
            id_values = []
            for id_field_name in self.__config.get_id_field_name():
                id_values.append(raw_content[id_field_name])
            content_id = id_merger(id_values)

            interfaces = self.__config.get_interfaces()
            for interface in interfaces:
                interface.new_content()
                interface.new_field(CONTENT_ID, content_id)

            # produce
            content = Content(content_id)
            for field_name in self.__config.get_field_name_list():
                pipeline_list = self.__config.get_pipeline_list(field_name)

                # search for timestamp override on specific field
                if type(raw_content[field_name]) == list:
                    timestamp = raw_content[field_name][1]
                    field_data = raw_content[field_name][0]
                else:
                    field_data = raw_content[field_name]

                # serialize for explanation
                memory_interface = self.__config.get_memory_interface(field_name)
                if memory_interface is not None:
                    memory_interface.new_field(field_name, field_data)

                # produce representations
                field = ContentField(field_name, timestamp)

                for i, pipeline in enumerate(pipeline_list):
                    content_technique = pipeline.get_content_technique()
                    if isinstance(content_technique, CollectionBasedTechnique):
                        field.append(content_technique.produce_content(str(i), content_id, field_name, str(pipeline)))
                    elif isinstance(content_technique, SingleContentTechnique):
                        preprocessor_list = pipeline.get_preprocessor_list()
                        processed_field_data = field_data
                        for preprocessor in preprocessor_list:
                            processed_field_data = preprocessor.process(processed_field_data)

                        field.append(content_technique.produce_content(str(i), processed_field_data))

                content.append(field)

            return content

    def __str__(self):
        return "ContentsProducer"

    def __repr__(self):
        msg = "< " + "ContentsProducer:" + "" \
                                           "config = " + str(self.__config) + " >"
        return msg
