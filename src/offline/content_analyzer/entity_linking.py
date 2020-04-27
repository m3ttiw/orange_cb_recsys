from offline.content_analyzer.field_content_production_technique import EntityLinking


class BabelPyEntityLinking(EntityLinking):
    """
    Interface for the Babelpy library that wrap some feature of Babelfy entity Linking.
    """
    def __init__(self):
        super().__init__()

    def produce_content(self, field_data):
        print("Entity Linking...")