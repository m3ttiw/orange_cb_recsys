from textblob import TextBlob

from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    SentimentalAnalysis


class TextBlobSentimentalAnalysis(SentimentalAnalysis):
    def __init__(self):
        super().__init__()

    def calculate_score(self):
        pass
