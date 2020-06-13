from babelpy.babelfy import BabelfyClient

from orange_cb_recsys.content_analyzer.content_representation.content_field import FeaturesBagField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    EntityLinking
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized


class BabelPyEntityLinking(EntityLinking):
    """
    Interface for the Babelpy library that wraps some feature of Babelfy entity Linking.
    """

    def __init__(self, api_key: str = None):
        super().__init__()
        self.__api_key = api_key
        self.__babel_client = None

    def set_lang(self, lang: str):
        super().set_lang(lang)
        params = dict()
        params['lang'] = self.get_lang()
        self.__babel_client = BabelfyClient(self.__api_key, params)

    def __str__(self):
        return "BabelPyEntityLinking"

    def produce_content(self, field_representation_name: str, field_data) -> FeaturesBagField:
        """
        Produces the field content for this representation
        Args:
            field_representation_name (str): Name of the field representation
            field_data: Data to use the produce the field content

        Returns:
            feature_bag (FeaturesBagField)
        """
        field_data = check_not_tokenized(field_data)

        self.__babel_client.babelfy(field_data)
        feature_bag = FeaturesBagField(field_representation_name)
        if self.__babel_client.entities is not None:
            for entity in self.__babel_client.entities:
                feature_bag.append_feature(entity['babelSynsetID'], entity['globalScore'])

        return feature_bag
