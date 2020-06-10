import lzma
import os
import pickle
import re

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.utils.const import logger


def load_content_instance(directory, content_id):
    logger.info("Loading %s" % content_id)
    try:
        content_filename = os.path.join(directory, content_id + '.xz')
        with lzma.open(content_filename, "r") as content_file:
            content: Content = pickle.load(content_file)
        return content
    except FileNotFoundError:
        return None


def get_unrated_items(items_directory, ratings):
    directory_filename_list = [os.path.splitext(filename)[0]
                               for filename in os.listdir(items_directory)
                               if filename != 'search_index']

    logger.info("Getting filenames from IDs")
    # list of id of item without rating
    rated_items_filename_list = set([re.sub(r'[^\w\s]', '', item_id) for item_id in ratings.to_id])

    logger.info("Checking if unrated")
    item_to_predict_id_list = [item_id for item_id in directory_filename_list if
                               item_id not in rated_items_filename_list]

    logger.info("Loading unrated items")
    item_to_predict_list = [
        load_content_instance(items_directory, item_id)
        for item_id in item_to_predict_id_list]

    return item_to_predict_list


def get_rated_items(items_directory, ratings):
    directory_filename_list = [os.path.splitext(filename)[0]
                               for filename in os.listdir(items_directory)
                               if filename != 'search_index']

    logger.info("Getting filenames from IDs")
    # list of id of item without rating
    rated_items_filename_list = set([re.sub(r'[^\w\s]', '', item_id) for item_id in ratings.to_id])

    logger.info("Checking if rated")
    item_to_predict_id_list = [item_id for item_id in directory_filename_list if item_id in rated_items_filename_list]

    item_to_predict_id_list.sort()

    logger.info("Loading rated items")
    item_to_predict_list = [
        load_content_instance(items_directory, item_id) for item_id in item_to_predict_id_list]

    return item_to_predict_list


def remove_not_existent_items(ratings, items_directory):
    directory_filename_list = [os.path.splitext(filename)[0]
                               for filename in os.listdir(items_directory)
                               if filename != 'search_index']

    rated_items_filename_list = set([re.sub(r'[^\w\s]', '', item_id) for item_id in ratings.to_id])

    intersection = [x for x in rated_items_filename_list if x in directory_filename_list]
    ratings = ratings[ratings["to_id"].isin(intersection)]

    return ratings
