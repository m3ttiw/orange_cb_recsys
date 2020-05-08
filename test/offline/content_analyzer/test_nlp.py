from unittest import TestCase

from src.offline.content_analyzer.nlp import NLTK


class TestNLTK(TestCase):
    """nltk = NLTK(stopwords_removal=True,
                stemming=True,
                lemmatization=True,
                named_entity_recognition=True,
                strip_multiple_whitespaces=True,
                url_tagging=True,
                lan="english")"""

    def test_process(self):
        self.skipTest("FIX TEST")
        #Test for only stop words removal
        nltka = NLTK(stopwords_removal=True, url_tagging=True)
        self.assertEqual(nltka.process(
                "The striped bats are hanging on their feet for the best"),
                "The striped bats hanging feet best")

        #Test for only stemming
        nltka.set_stemming(True)
        nltka.set_stopwords_removal(False)
        self.assertEqual(nltka.process(
                "My name is Francesco and I am a student at the University of the city of Bari"),
                "my name is francesco and i am a student at the univers of the citi of bari")
        nltka.set_stemming(False)

        #Test for only lemmatization
        nltka.set_lemmatization(True)
        self.assertEqual(nltka.process(
                "The striped bats are hanging on their feet for best"),
                "The strip bat be hang on their foot for best")

        #Test for lemmatization with multiple whitespaces removal
        nltka.set_strip_multiple_whitespaces(True)
        self.assertEqual(nltka.process(
                "The   striped  bats    are    hanging   on   their    feet   for  best"),
                "The strip bat be hang on their foot for best")

        #Test for lemmatization with multiple whitespaces removal and URL tagging
        nltka.set_url_tagging(True)
        self.assertEqual(nltka.process(
                "The   striped http://facebook.com bats https://github.com   are   http://facebook.com hanging   on   their    feet   for  best  http://twitter.it"),
                "The strip <URL> bat <URL> be <URL> hang on their foot for best <URL>")

        # Test for lemmatization, multiple whitespaces removal, URL tagging and stemming
        nltka.set_stemming(True)
        self.assertEqual(nltka.process(
            "The   striped http://facebook.com bats https://github.com   are   http://facebook.com hanging   on   their    feet   for  best  http://twitter.it"),
            "the strip <url> bat <url> be <url> hang on their foot for best <url>")

        # Test for lemmatization, multiple whitespaces removal, URL tagging, stemming, stop words removal
        nltka.set_stopwords_removal(True)
        self.assertEqual(nltka.process(
            "The   striped http://facebook.com bats https://github.com   are   http://facebook.com hanging   on   their    feet   for  best  http://twitter.it"),
            "the strip < url > bat < url > < url > hang foot best < url >")
