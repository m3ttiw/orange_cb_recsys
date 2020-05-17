from typing import List
import pickle
from src.offline.content_analyzer.content_representation.content_field import ContentField


class Content:
    """
    Class that represents a content. A content can be an item, a user or a rating
    A content is identified by a string id and is composed of different fields
    Args:
        content_id (str): identifier
        field_list (list[ContentField]): list of the fields instances of a content
    """
    def __init__(self, content_id: str,
                 field_list: List[ContentField] = None):
        if field_list is None:
            field_list = []         # list o dict
        self.__content_id: str = content_id
        self.__field_list: List[ContentField] = field_list

    def get_field_list(self):
        return self.__field_list

    def append(self, field: ContentField):
        self.__field_list.append(field)

    def remove(self, field_name: str):
        """
        Remove the field named field_name from the field list
        Args:
            field_name (str): the name of the field to remove
        """

        i = 0
        for field in self.__field_list:
            if field.get_name() == field_name:
                break
            i += 1

        self.__field_list.pop(i)

    def serialize(self, output_directory: str):
        """
        Serialize a content instance
        """
        with open(output_directory + '/' + self.__content_id + '.bin', 'wb') as file:
            pickle.dump(self, file)

    def __str__(self):
        content_string = "Content:" + self.__content_id
        field_string = ""
        for field in self.__field_list:
            field_string += str(field) + "\n"

        return content_string + '\n\n' + field_string + "##############################"

    def __eq__(self, other):
        return self.__content_id == other.__content_id and self.__field_list == other.__field_list


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
