from textblob import TextBlob

from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    SentimentalAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource


class TextBlobSentimentalAnalysis(SentimentalAnalysis):
    """
    Interface for the textblob library that does sentimental analysis on text.
    """

    def __init__(self, field_name: str,
                 source: RawInformationSource):
        super().__init__()
        self.__field_name = field_name
        self.__source = source

    def __str__(self):
        return "TextBlobSentimentalAnalysis"

    def calculate_score(self):
        pass
