from src.offline.raw_data_extractor.information_serializer import InformationSerializer
from src.offline.raw_data_extractor.raw_information_source import RawInformationSource


class RawFieldPipeline: # passaggi per estrarre e serializzare contenuto di un field
    def __init__(self, field_source: RawInformationSource,
                 field_serializer: InformationSerializer):
        self.__field_source: RawInformationSource = field_source
        self.__field_serializer: InformationSerializer = field_serializer

    def get_field_source(self):
        return self.__field_source

    def get_field_serializer(self):
        return self.__field_serializer

    def set_field_source(self, field_source: RawInformationSource):
        self.__field_source = field_source

    def set_field_serializer(self, field_serializer: InformationSerializer):
        self.__field_serializer = field_serializer


class Config:
    def __init__(self, fields_pipeline: dict = None):
        if fields_pipeline is None:
            fields_pipeline = {}
        self.__fields_pipeline: dict = fields_pipeline

    def add_pipeline(self, field_name: str, field_pipeline: RawFieldPipeline):
        self.__fields_pipeline[field_name] = field_pipeline

    def get_pipeline(self, field_name: str):
        return self.__fields_pipeline[field_name]

    def get_field_names(self):
        return self.__fields_pipeline.keys()


class RawDataManager:
    def __init__(self, item_id_list: list,
                 config: Config):
        self.__item_id_list: list = item_id_list
        self.__config: Config = config

    def start(self):
        field_names = self.__config.get_field_names()

        for item_id in self.__item_id_list:
            for field_name in field_names:
                field_source = self.__config.get_pipeline(field_name).get_field_source()
                field_data = field_source.extract_field_data(item_id, field_name)
                field_serializer = self.__config.get_pipeline(field_name).get_field_serializer()
                field_serializer.serialize(field_data)

