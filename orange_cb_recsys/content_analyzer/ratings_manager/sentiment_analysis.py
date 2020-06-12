from textblob import TextBlob

from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import SentimentAnalysis


class TextBlobSentimentAnalysis(SentimentAnalysis):
    """
    Interface for the textblob library that does sentimental analysis on text.
    """

    def __str__(self):
        return "TextBlobSentimentalAnalysis"

    def __repr__(self):
        return "< TextBlobSentimentalAnalysis >"

    def fit(self, field_data: str) -> float:
        """
        This method calculates the sentiment analysis score on textual reviews
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
        return TextBlob(field_data).sentiment.polarity
