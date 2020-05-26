from typing import List, Dict
import pickle
import re

from orange_cb_recsys.content_analyzer.content_representation.content_field import ContentField


class Content:
    """
    Class that represents a content. A content can be an item, a user or a rating
    A content is identified by a string id and is composed of different fields
    Args:
        content_id (str): identifier
        field_dict (list[ContentField]): list of the fields instances of a content
    """
    def __init__(self, content_id: str,
                 field_dict: Dict[str, ContentField] = None):
        if field_dict is None:
            field_dict = {}       # list o dict
        self.__content_id: str = content_id
        self.__field_dict: Dict[str, ContentField] = field_dict

    def get_field_list(self):
        return self.__field_dict

    def get_field(self, field_name: str):
        return self.__field_dict[field_name]

    def append(self, field_name: str, field: ContentField):
        self.__field_dict[field_name] = field

    def remove(self, field_name: str):
        """
        Remove the field named field_name from the field list
        Args:
            field_name (str): the name of the field to remove
        """

        self.__field_dict.pop(field_name)

    def serialize(self, output_directory: str):
        """
        Serialize a content instance
        """
        file_name = re.sub(r'[^\w\s]','', self.__content_id)
        with open(output_directory + '/' + file_name + '.bin', 'wb') as file:
            pickle.dump(self, file)

    def __str__(self):
        content_string = "Content:" + self.__content_id
        field_string = ""
        for field in self.__field_dict:
            field_string += str(field) + "\n"

        return content_string + '\n\n' + field_string + "##############################"

    def get_content_id(self):
        return self.__content_id

    def __eq__(self, other):
        return self.__content_id == other.__content_id and self.__field_dict == other.__field_dict


class RepresentedContents:
    """
    Class that collects the Contents instance created,
    the whole collection can be serialized.
    Args:
        length: number of contents
        content_list (list<Content>): list of content's instances
    """
    def __init__(self, length: int = 0,
                 content_list: List[Content] = None):
        if content_list is None:
            content_list = []
        self.__content_list: List[Content] = content_list
        self.__length: int = length

    def append(self, content: Content):
        self.__content_list.append(content)

    def serialize(self):
        """
        Serialize the entire collection
        Returns:

        """

    def __str__(self):
        return str(self.__content_list)
