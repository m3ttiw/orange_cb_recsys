from typing import Dict
import time
import os
from os import path

from orange_cb_recsys.content_analyzer.config import ContentAnalyzerConfig, FieldRepresentationPipeline
from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.content_analyzer.content_representation.content_field import ContentField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import CollectionBasedTechnique, \
    SingleContentTechnique
from orange_cb_recsys.utils.id_merger import id_merger

parent_dir = '../../contents'
if not path.exists(parent_dir):
    parent_dir = 'contents'


class ContentAnalyzer:
    """
    Class to whom the control of the content analysis phase is delegated,

    Args:
        config (ContentAnalyzerConfig):
            configuration for processing the item fields. This parameter provides the possibility
            of customizing the way in which the input data is processed.
    """

    def __init__(self, config: ContentAnalyzerConfig):
        self.__config: ContentAnalyzerConfig = config

    def set_config(self, config: ContentAnalyzerConfig):
        self.__config = config

    def __dataset_refactor(self):
        for field_name in self.__config.get_field_name_list():
            for pipeline in self.__config.get_pipeline_list(field_name):
                technique = pipeline.get_content_technique()
                if isinstance(technique, CollectionBasedTechnique):
                    technique.set_field_need_refactor(field_name)
                    technique.set_pipeline_need_refactor(str(pipeline))
                    technique.set_processor_list(pipeline.get_preprocessor_list())
                    technique.dataset_refactor(self.__config.get_source(), self.__config.get_id_field_name())

    def fit(self):
        """
        Begins to process the creation of the contents

        Returns:
            List<Content>:
                list which elements are the produced content instances
        """

        output_path = os.path.join(parent_dir, self.__config.get_output_directory())
        os.mkdir(output_path)

        contents_producer = ContentsProducer.get_instance()
        contents_producer.set_config(self.__config)

        interfaces = self.__config.get_interfaces()
        for interface in interfaces:
            interface.init_writing()

        self.__dataset_refactor()

        for raw_content in self.__config.get_source():
            content = contents_producer.create_content(raw_content)
            print(content)
            content.serialize(output_path)

        for interface in interfaces:
            interface.stop_writing()

    def __str__(self):
        return "ContentAnalyzer"

    def __repr__(self):
        msg = "< " + "ContentAnalyzer: " + "" \
                                           "config = " + str(self.__config) + "; >"
        return msg


class ContentsProducer:
    """
    Singleton class which encapsulates the creation process of the items,
    The creation process is specified in the config parameter of ContentAnalyzer and
    is supposed to be the same for each item.
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

    def __get_timestamp(self, raw_content: Dict) -> str:
        # search for timestamp as dataset field, no timestamp needed for items
        timestamp = None
        if self.__config.get_content_type() != "item":
            if "timestamp" in raw_content.keys():
                timestamp = raw_content["timestamp"]
            else:
                timestamp = time.time()

        return timestamp

    def __create_field(self, raw_content: Dict, field_name: str, content_id: str, timestamp: str):
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

        for i, pipeline in enumerate(self.__config.get_pipeline_list(field_name)):
            if isinstance(pipeline.get_content_technique(), CollectionBasedTechnique):
                field.append(str(i), self.__create_representation_CBT(str(i), field_name, content_id, pipeline))
            elif isinstance(pipeline.get_content_technique(), SingleContentTechnique):
                field.append(str(i), self.__create_representation(str(i), field_data, pipeline))
            elif pipeline.get_content_technique() is None:
                field.append(str(i), field_data)

        return field

    @staticmethod
    def __create_representation_CBT(field_representation_name: str, field_name: str, content_id: str, pipeline: FieldRepresentationPipeline):
        return pipeline.get_content_technique().\
            produce_content(field_representation_name, content_id, field_name, str(pipeline))

    @staticmethod
    def __create_representation(field_representation_name: str, field_data, pipeline: FieldRepresentationPipeline):
        preprocessor_list = pipeline.get_preprocessor_list()
        processed_field_data = field_data
        for preprocessor in preprocessor_list:
            processed_field_data = preprocessor.process(processed_field_data)

        return pipeline.get_content_technique().\
            produce_content(field_representation_name, processed_field_data)

    def create_content(self, raw_content: Dict):
        """
        Creates a content processing every field in the specified way.
        This class is iteratively invoked by the fit method.

        Returns:
            Content: an instance of content with his fields

        Raises:
            general Exception
        """

        if self.__config is None:
            raise Exception("You must set a config with set_config()")
        else:
            CONTENT_ID = "content_id"

            timestamp = self.__get_timestamp(raw_content)

            # construct id from the list of the fields that compound id
            content_id = id_merger(raw_content, self.__config.get_id_field_name())

            interfaces = self.__config.get_interfaces()
            for interface in interfaces:
                interface.new_content()
                interface.new_field(CONTENT_ID, content_id)

            # produce
            content = Content(content_id)
            for field_name in self.__config.get_field_name_list():
                # search for timestamp override on specific field
                content.append(field_name, self.__create_field(raw_content, field_name, content_id, timestamp))

            for interface in interfaces:
                interface.serialize_content()

            return content

    def __str__(self):
        return "ContentsProducer"

    def __repr__(self):
        msg = "< " + "ContentsProducer:" + "" \
                                           "config = " + str(self.__config) + " >"
        return msg
