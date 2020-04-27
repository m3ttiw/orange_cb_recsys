def extract_ids(filepath: str, id_index: int = 0, separator: str = "::", start_index: int = 0, stop_index=-1):
    with open(filepath) as f:
        item_id_list = []

        i = 1
        for line in f:
            if stop_index != -1 and i == stop_index:
                break

            if i >= start_index:
                item_id_list.append(line.split(separator)[id_index])

            i += 1

    return item_id_list
