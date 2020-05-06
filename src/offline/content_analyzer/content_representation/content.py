from typing import List
from src.offline.content_analyzer.content_representation.content_field import ContentField


class Content:
    """
    Class that represent a content,
    a content can be an item, a user or a rating
    A content is identified by a string id and is composed of different fields
    Args:
        content_id (str): identifier
        fields (list[ContentField]): list of the fields instances of a content
    """
    def __init__(self, content_id: str, fields: List[ContentField] = None):
        if fields is None:
            fields = []         # list o dict
        self.__content_id: str = content_id
        self.__fields: List[ContentField] = fields

    def append(self, field: ContentField):
        self.__fields.append(field)

    def remove(self, field_name: str):
        """
        remove the field with field_name in the fields list
        Args:
            field_name (str): the name of the field to remove
        """
        self.__fields.pop(self.__fields.index(ContentField(field_name)))

    def serialize(self):
        """
        Serialize an item
        """


class RepresentedContents:
    """
    Class that collects the Contents instance created,
    the whole collection can be serialized.
    Args:
        contents (list<Item>): list of content's instances
        length: number of contents
    """
    def __init__(self, length: int = 0, contents: List[Content] = None):
        if contents is None:
            contents = []
        self.__contents: List[Content] = contents
        self.__length: int = length

    def append(self, content: Content):
        self.__contents.append(content)

    def serialize(self):
        """
        Serialize the entire collection
        Returns:

        """