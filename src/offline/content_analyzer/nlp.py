from abc import ABC

from offline.content_analyzer.information_processor import NLP


class OpenNLP(NLP):
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
