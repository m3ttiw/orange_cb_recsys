import lzma
import os
import pickle
import re

from orange_cb_recsys.content_analyzer.content_representation.content import Content


def load_content_instance(directory, content_id):
    content_filename = os.path.join(directory, content_id + '.xz')
    with lzma.open(content_filename, "r") as content_file:
        content: Content = pickle.load(content_file)
    return content


def get_unrated_items(items_directory, ratings):
    directory_file_list = [os.path.splitext(filename)[0]
                           for filename in os.listdir(items_directory)
                           if filename != 'search_index']

    # list of item id
    directory_item_id_list = [
        load_content_instance(items_directory, item_filename).get_content_id() for
        item_filename in directory_file_list]

    # list of id of item without rating
    item_to_predict_id_list = [item_id for item_id in directory_item_id_list if
                               not ratings['to_id'].str.contains(item_id).any()]

    item_to_predict_list = [
        load_content_instance(items_directory, re.sub(r'[^\w\s]', '', item_id))
        for item_id in item_to_predict_id_list]

    return item_to_predict_list


def get_rated_items(items_directory, ratings):
    directory_file_list = [os.path.splitext(filename)[0]
                           for filename in os.listdir(items_directory)
                           if filename != 'search_index']

    # list of item id
    directory_item_id_list = [
        load_content_instance(items_directory, item_filename).get_content_id() for
        item_filename in directory_file_list]

    # list of id of item without rating
    item_to_predict_id_list = [item_id for item_id in directory_item_id_list if
                               ratings['to_id'].str.contains(item_id).any()]

    item_to_predict_list = [
        load_content_instance(items_directory, re.sub(r'[^\w\s]', '', item_id))
        for item_id in item_to_predict_id_list]

    return item_to_predict_list
