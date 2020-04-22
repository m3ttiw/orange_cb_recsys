from src.offline.content_analyzer.field_content_production_technique import FieldContentProductionTechnique


class ContentAnalyzer:
    def __init__(self, field_content_technique: dict = None, field_preprocessing: dict = None):
        if field_content_technique is None:
            field_content_technique = {}
        if field_preprocessing is None:
            field_preprocessing = {}
        self.__field_content_technique = field_content_technique
        self.__field_preprocessing = field_preprocessing

    def add_content_technique(self, technique):
        pass

    def add_preprocessing_technique(self, technique):
        pass


class ItemProducer:
    def __init__(self, field_content_technique: dict, field_preprocessing: dict):
        self.__field_content_technique = field_content_technique
        self.__field_preprocessing = field_preprocessing

    def create_item(self, item_id: str):
        pass
