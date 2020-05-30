from textblob import TextBlob

from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    SentimentalAnalysis


class TextBlobSentimentalAnalysis(SentimentalAnalysis):
    """
    Interface for the textblob library that does sentimental analysis on text.
    """
    def __init__(self):
        super().__init__()

    def calculate_score(self):
        pass
