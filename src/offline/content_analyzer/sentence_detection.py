import nltk

from typing import List
from nltk.tokenize import sent_tokenize

from src.offline.content_analyzer.field_content_production_technique \
    import SentenceDetectionTechnique


class NLTKSentenceDetection(SentenceDetectionTechnique):
    """
    Implements abstract class SentenceDetectionTechnique,
    in this class nltk library is used for this operation
    """
    def detect_sentences(self, text: str) -> List[str]:
        """
        Implements the abstract method using the nltk library

        Args:
            text (str): text that will be divided

        Returns:
            List<str>: list of sentences
        """

        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt')

        sentences = sent_tokenize(text)
        for i, sentence in enumerate(sentences):
            sentences[i] = sentence[:len(sentence) - 1]

        return sentences
