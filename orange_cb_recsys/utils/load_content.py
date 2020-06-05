import pickle

from orange_cb_recsys.content_analyzer.content_representation.content import Content


def load_content_instance(directory, content_id):
    content_filename = directory + '/' + content_id + '.bin'
    with open(content_filename, "rb") as content_file:
        content: Content = pickle.load(content_file)

    return content
