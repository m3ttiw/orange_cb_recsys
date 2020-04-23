from src.offline.content_analyzer.field_content_production_technique import FieldContentProductionTechnique
from src.offline.content_analyzer.item_representation.item import Item
from src.offline.content_analyzer.item_representation.item_field import ItemField


class Config:
    def __init__(self, fields_content_technique: dict = None,
                 fields_preprocessing: dict = None,
                 fields_loader: dict = None):
        if fields_content_technique is None:
            fields_content_technique = {}
        if fields_preprocessing is None:
            fields_preprocessing = {}
        if fields_loader is None:
            fields_loader = {}
        self.__fields_content_technique = fields_content_technique
        self.__fields_preprocessing = fields_preprocessing
        self.__fields_loader = fields_loader

    def add_content_technique(self, field_name: str, technique):
        self.__fields_content_technique[field_name].append(technique)       # lista di tecniche

    def add_preprocessing_technique(self, field_name: str, technique):
        pass

    def add_loader(self, field_name: str, loader):
        pass

    def get_field_names(self):
        return self.__fields_content_technique.keys()

    def get_content_technique(self, field_name):
        return self.__fields_content_technique[field_name]

    def get_preprocessing(self, field_name):
        return self.__fields_preprocessing[field_name]

    def get_loader(self, field_name):
        return self.__fields_loader[field_name]


class ContentAnalyzer:
    def __init__(self, item_id_list: list,
                 config: Config):
        self.__item_id_list = item_id_list
        self.__config = config

    def set_config(self, config: Config):
        self.__config = config

    def start(self):
        items_producer = ItemsProducer.get_instance().set_config(self.__config)
        for item_id in self.__item_id_list:
            item = items_producer.create_item(item_id)


class ItemsProducer:
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ItemsProducer.__instance is None:
            ItemsProducer()
        return ItemsProducer.__instance

    def __init__(self):
        self.__config = None
        """ Virtually private constructor. """
        if ItemsProducer.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ItemsProducer.__instance = self

    def set_config(self, config: Config):
        self.__config = config

    def create_item(self, item_id: str):
        if self.__config is None:
            raise Exception("You must set a config with set_config()")
        else:
            field_name_list = self.__config.get_field_names()
            item = Item(item_id)
            for field_name in field_name_list:
                loader = self.__config.get_loader(field_name)
                field_data = loader.load(item_id, field_name)
                # possono esserci più preprocessor per field
                preprocessor = self.__config.get_preprocessor(field_name)
                if preprocessor is not None:
                    field_data = preprocessor.process(field_data)
                content_techniques = self.__config.get_content_technique(field_name)
                field = ItemField(field_name)
                for content_technique in content_techniques:
                    field.append(content_technique.produce_content(field_data))
                item.append(field)




