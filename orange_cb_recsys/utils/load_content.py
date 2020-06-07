import lzma
import os
import pickle

from orange_cb_recsys.content_analyzer.content_representation.content import Content


def load_content_instance(directory, content_id):
    content_filename = os.path.join(directory, content_id + '.xz')
    with lzma.open(content_filename, "r") as content_file:
        content: Content = pickle.load(content_file)
    return content
