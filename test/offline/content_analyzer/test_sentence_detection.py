from unittest import TestCase

from src.content_analyzer.field_content_production_techniques import NLTKSentenceDetection


class TestNLTKSentenceDetection(TestCase):
    def test_detect_sentences(self):
        text = "god is great! i won lottery."
        detector = NLTKSentenceDetection()

        expected = ["god is great", "i won lottery"]
        result = detector.detect_sentences(text)

        self.assertEqual(result, expected)
