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
        super().__init__(field_name, source)

    def __str__(self):
        return "TextBlobSentimentalAnalysis"

    def __repr__(self):
        return "< TextBlobSentimentalAnalysis :" + \
               "source = " + str(self.get_source())

    def calculate_score(self):
        for line in self.__source:
            if type(self.__field_name) == str:
                return TextBlob(line[self.__field_name]).sentiment.polarity
            raise TypeError("field_name should contain a string")
