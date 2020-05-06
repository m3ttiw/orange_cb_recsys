from offline.content_analyzer.field_content_production_technique import SentenceDetectionTechnique
from nltk.tokenize import sent_tokenize


class NLTKSentenceDetection(SentenceDetectionTechnique):
    def __init__(self):
        super().__init__()

    def detect_sentences(self, text: str):
        sentences = sent_tokenize(text)
        for i, sentence in enumerate(sentences):
            sentences[i] = sentence[:len(sentence) - 1]

        return sentences
