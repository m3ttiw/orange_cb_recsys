from src.content_analyzer.content_representation.content_field import FeaturesBagField
from babelpy.babelfy import BabelfyClient

from src.content_analyzer.field_content_production_techniques.field_content_production_technique import EntityLinking
from src.utils.check_tokenization import check_not_tokenized


class BabelPyEntityLinking(EntityLinking):
    """
    Interface for the Babelpy library that wraps some feature of Babelfy entity Linking.
    """

    def __init__(self, lang: str = "EN", api_key: str = None):
        super().__init__()
        params = dict()
        params['lang'] = lang
        self.__babel_client = BabelfyClient(api_key, params)

    def __str__(self):
        return "BabelPyEntityLinking"

    def produce_content(self, field_representation_name: str, field_data: str) -> FeaturesBagField:
        field_data = check_not_tokenized(field_data)

        self.__babel_client.babelfy(field_data)
        feature_bag = FeaturesBagField(field_representation_name)
        if self.__babel_client.entities is not None:
            for entity in self.__babel_client.entities:
                feature_bag.append_feature(entity['babelSynsetID'], entity['globalScore'])
        return feature_bag
