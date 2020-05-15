def id_merger(id_list):
    """
    This method ise used to compact a list of ids into a unique string. This can be useful when
    there is content whose id is composed by values coming from more than one field.

    Args:
        id_list (List<str>): List containing one or more ids

    Returns:
        id_merged (str): String in which the values contained in the list given in input are
            merged
    """
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
