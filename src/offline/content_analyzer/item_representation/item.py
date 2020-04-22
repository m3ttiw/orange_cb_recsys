from src.offline.content_analyzer.item_representation.item_field import ItemField


class RepresentedItems:
    def __init__(self, items, length):
        self.__items = items
        self.__length = length

    def serialize(self):
        pass                # forse non necessario (?)


class Item:
    def __init__(self, item_id: str, fields: list = None):
        if fields is None:
            fields = []         # list o dict
        self.__id = item_id
        self.__fields = fields

    def add(self, field: ItemField):
        self.__fields.append(field)

    def remove(self, field_name: str):
        self.__fields.pop(self.__fields.index(ItemField(field_name)))

    def serialize(self):
        pass
