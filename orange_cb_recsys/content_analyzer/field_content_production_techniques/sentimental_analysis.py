from textblob import TextBlob

from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    SentimentalAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource


class TextBlobSentimentalAnalysis(SentimentalAnalysis):
    """
    Interface for the textblob library that does sentimental analysis on text.

    Args:
        field_name (str): the name of the field with the textual reviews
        source (RawInformationSource): source file with the reviews
    """

    def __init__(self, field_name: str,
                 source: RawInformationSource):
        super().__init__(field_name, source)

    def __str__(self):
        return "TextBlobSentimentalAnalysis"

    def __repr__(self):
        return "< TextBlobSentimentalAnalysis :" + \
               "source = " + str(self.get_source())

    def calculate_score(self) -> list:
        """
        This method calculate the sentiment analysis score on textual reviews
        Returns:
            sentiment_data: a list of sentiment analysis score
        """
        sentiment_data = list()
        for line in self.get_source():
            if type(self.get_field_name()) == str:
                text = TextBlob(line[self.get_field_name()]).sentiment.polarity
                sentiment_data.append(text)
        return sentiment_data
