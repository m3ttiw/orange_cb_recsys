from babelpy.config.config import LANG, API_KEY

from src.offline.content_analyzer.content_representation.content_field import FeaturesBagField
from src.offline.content_analyzer.field_content_production_technique import EntityLinking
from babelpy.babelfy import BabelfyClient


class BabelPyEntityLinking(EntityLinking):
    """
    Interface for the Babelpy library that wrap some feature of Babelfy entity Linking.
    """

    def __init__(self, lang: str = "EN", api_key: str = None):
        super().__init__()
        params = dict()
        params['lang'] = lang
        self.__babel_client = BabelfyClient(api_key, params)

    def produce_content(self, field_representation_name: str, **kwargs) -> FeaturesBagField:
        self.__babel_client.babelfy(kwargs["field_data"])
        feature_bag = FeaturesBagField('repr_field_name')
        if self.__babel_client.entities is not None:
            for entity in self.__babel_client.entities:
                feature_bag.append_feature(entity['babelSynsetID'], entity['globalScore'])
        return feature_bag
