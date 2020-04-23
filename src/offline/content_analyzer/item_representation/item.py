from src.offline.content_analyzer.item_representation.item_field import ItemField


class RepresentedItems:
    def __init__(self, items: list, length: int):
        self.__items: list = items
        self.__length: int = length

    def serialize(self):
        pass                # forse non necessario (?)


class Item:
    def __init__(self, item_id: str, fields: list = None):
        if fields is None:
            fields = []         # list o dict
        self.__id: str = item_id
        self.__fields: list = fields

    def append(self, field: ItemField):
        self.__fields.append(field)

    def remove(self, field_name: str):
        self.__fields.pop(self.__fields.index(ItemField(field_name)))

    def serialize(self):
        pass
