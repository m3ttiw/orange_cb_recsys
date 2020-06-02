from textblob import TextBlob

from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import SentimentalAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource


class TextBlobSentimentalAnalysis(SentimentalAnalysis):
    """
    Interface for the textblob library that does sentimental analysis on text.

    Args:
        field_name (str): the name of the field with the textual reviews
    """

    def __init__(self, field_name: str):
        super().__init__(field_name)

    def __str__(self):
        return "TextBlobSentimentalAnalysis"

    def __repr__(self):
        return "< TextBlobSentimentalAnalysis: field_name = {}>".format(self.get_field_name())

    def fit(self, field_data: object) -> float:
        """
        This method calculate the sentiment analysis score on textual reviews
        Returns:
            sentiment_data: a list of sentiment analysis score
        """

        """
        sentiment_data = list()         ### RIADATTARE A UN SOLO FIELD
        for line in self.get_source():
            if type(self.get_field_name()) == str:
                text = TextBlob(line[self.get_field_name()]).sentiment.polarity
                sentiment_data.append(text)
        return sentiment_data
        """
        try:
            self.__type_check(field_data)
            score = TextBlob(field_data).sentiment.polarity
        except TypeError:
            print("TypeError: Sentiment analisys does not work on this field_data={}".format(field_data))
            score = None
        return score
