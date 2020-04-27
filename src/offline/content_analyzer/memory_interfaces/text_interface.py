from offline.content_analyzer.memory_interfaces.memory_interfaces import TextInterface


class IndexInterface(TextInterface):
    """
    Abstract class that serialize and load text in a index structure
    """
    def __init__(self, directory: str):
        super().__init__(directory)

    def serialize(self, field_data):
        print("index writing")

    def load(self, item_id: str, field_name: str):
        print("index reading")
