from abc import ABC

from offline.content_analyzer.information_processor import NLP


class OpenNLP(NLP):
    """
    Interface for the library Opennlp for natural language processing features

    Args:
        stopwords_removal (bool): Whether you want to remove stop words
        stemming (bool): Whether you want to execute stemming
        lemmatization (bool):  Whether you want to execute lemmatization
        named_entity_recognition (bool): Whether you want to execute named entity recognition
        strip_multiple_whitespaces (bool): Whether you want to remove multiple whitespaces
        url_tagging (bool): Whether you want to tag the urls in the text and to replace with "<URL>"
    """
    def __init__(self, stopwords_removal: bool = False,
                 stemming: bool = False,
                 lemmatization: bool = False,
                 named_entity_recognition: bool = False,
                 strip_multiple_whitespaces: bool = True,
                 url_tagging: bool = False):

        super().__init__(stopwords_removal,
                         stemming, lemmatization, named_entity_recognition,
                         strip_multiple_whitespaces, url_tagging)

    def process(self, field_data):
        print("text processing using OpenNLP library")
