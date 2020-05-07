from typing import List


def id_merger(id_list):
    if type(id_list) == str or type(id_list) == int:
        return str(id_list)
    elif type(id_list) == list:
        id_merged = ""
        for i in range(len(id_list)):
            id_merged += str(id_list[i])
            if i != len(id_list) - 1:
                id_merged += "_"
        return id_merged
    else:
        raise TypeError("id must be an integer, a string or a list of strings and/or integer")
