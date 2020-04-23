from src.offline.content_analyzer.field_content_production_technique import FieldContentProductionTechnique
from src.offline.content_analyzer.information_loader import InformationLoader
from src.offline.content_analyzer.information_processor import InformationProcessor
from src.offline.content_analyzer.item_representation.item import Item
from src.offline.content_analyzer.item_representation.item_field import ItemField


class FieldContentPipeline:
    def __init__(self, loader: InformationLoader,
                 content_technique: FieldContentProductionTechnique,
                 preprocessor: InformationProcessor = None):
        self.__loader: InformationLoader = loader
        self.__preprocessor: InformationProcessor = preprocessor
        self.__content_technique: FieldContentProductionTechnique = content_technique

    def set_loader(self, loader: InformationLoader):
        self.__loader = loader

    def set_preprocessor(self, preprocessor: InformationProcessor):
        self.__preprocessor = preprocessor

    def set_content_technique(self, content_technique: FieldContentProductionTechnique):
        self.__content_technique = content_technique

    def get_laoder(self):
        return self.__loader

    def get_preprocessor(self):
        return self.__preprocessor

    def get_content_technique(self):
        return self.__content_technique


class Config:
    def __init__(self, field_content_pipeline: dict = None):
        if field_content_pipeline is None:
            field_content_pipeline = {}

        self.__field_content_pipeline: dict = field_content_pipeline

    def add_pipeline(self, field_name: str, pipeline: FieldContentPipeline):
        self.__field_content_pipeline[field_name].append(pipeline)

    def get_pipeline_list(self, field_name: str):
        return self.__field_content_pipeline[field_name]

    def get_field_names(self):
        return self.__field_content_pipeline.keys()


class ContentAnalyzer:
    def __init__(self, item_id_list: list,
                 config: Config):
        self.__item_id_list: list = item_id_list
        self.__config: Config = config

    def set_config(self, config: Config):
        self.__config = config

    def start(self):
        items_producer = ItemsProducer.get_instance().set_config(self.__config)
        items = []
        field_name_list = self.__config.get_field_names()
        for item_id in self.__item_id_list:
            items.append(items_producer.create_item(item_id, field_name_list))

        return items


class ItemsProducer:
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ItemsProducer.__instance is None:
            ItemsProducer()
        return ItemsProducer.__instance

    def __init__(self):
        self.__config: Config = None
        """ Virtually private constructor. """
        if ItemsProducer.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ItemsProducer.__instance = self

    def set_config(self, config: Config):
        self.__config = config

    def create_item(self, item_id: str, field_name_list: list):
        if self.__config is None:
            raise Exception("You must set a config with set_config()")
        else:
            item = Item(item_id)
            for field_name in field_name_list:
                pipeline_list = self.__config.get_pipeline_list(field_name)
                field = ItemField(field_name)
                for pipeline in pipeline_list:
                    loader = pipeline.get_loader()
                    field_data = loader.load(item_id, field_name)
                    preprocessor = pipeline.get_preprocessor()
                    if preprocessor is not None:
                        field_data = preprocessor.process(field_data)

                    content_technique = pipeline.get_content_technique()
                    field.append(content_technique.produce_content(field_data))
                item.append(field)

            return item
