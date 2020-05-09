import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from src.offline.content_analyzer.information_processor import NLP


def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


class NLTK(NLP):
    """
    Interface for the library OpenNlp for natural language processing features

    """
    def __init__(self, stopwords_removal: bool = False,
                 stemming: bool = False,
                 lemmatization: bool = False,
                 named_entity_recognition: bool = False,
                 strip_multiple_whitespaces: bool = True,
                 url_tagging: bool = False,
                 lan: str = "english"):

        super().__init__(stopwords_removal,
                         stemming, lemmatization, named_entity_recognition,
                         strip_multiple_whitespaces, url_tagging)
        self.__lan: str = lan
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')

    def get_lan(self) -> str:
        return self.__lan

    def set_lan(self, lan: str):
        self.__lan = lan

    def __tokenization_operation(self, text):
        self.set_is_tokenized(True)
        return word_tokenize(text)

    def __stopwords_removal_operation(self, text) -> str:
        """
        Execute stopwords removal on input text

        Args:
            text:

        Returns:
            str: input text without stopwords
        """
        stop_words = set(stopwords.words(self.get_lan()))

        filtered_sentence = []
        for word_token in text:
            if word_token not in stop_words:
                filtered_sentence.append(word_token)

        return filtered_sentence

    def __stemming_operation(self, text):
        """
        Execute stemming on input text

        Args:
            text:

        Returns:
            str: input text, words reduced to their stems
        """
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer(language=self.get_lan())

        stemmed_text = []
        for word in text:
            stemmed_text.append(stemmer.stem(word))

        return stemmed_text

    def __lemmatization_operation(self, text):
        """
        Execute lemmatization on input text

        Args:
            text:

        Returns:
            str: input text, words reduced to their lemma
        """

        lemmatizer = WordNetLemmatizer()
        lemmatized_text = []
        for word in text:
            lemmatized_text.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
        return lemmatized_text

    def __named_entity_recognition_operation(self, text):
        """
        Execute NER on input text

        Args:
            text:

        Returns:
            str: input text, entities recognized
        """
        text = nltk.pos_tag(text)
        pattern = 'NP: {<DT>?<JJ>*<NN>}'
        cp = nltk.RegexpParser(pattern)
        cs = cp.parse(text)
        return cs

    def __strip_multiple_whitespaces_operation(self, text):
        """
        Remove multiple whitespaces on input text

        Args:
            text:

        Returns:
            str: input text, multiple whitespaces removed
        """

        import re
        return re.sub(' +', ' ', text)

    @staticmethod
    def __url_tagging_operation(text):
        """
        substitute urls with <URL> string on input text

        Args:
            text:

        Returns:
            str: input text, <URL> instead of full url
        """
        import re
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]| '
                          '[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                          text)
        for url in urls:
            text = text.replace(url, "<URL>")
        return text

    def process(self, field_data):
        if self.get_strip_multiple_whitespaces():
            field_data = self.__strip_multiple_whitespaces_operation(field_data)
        if self.get_url_tagging():
            field_data = self.__url_tagging_operation(field_data)

        field_data = self.__tokenization_operation(field_data)
        if self.get_stopwords_removal():
            field_data = self.__stopwords_removal_operation(field_data)
        if self.get_lemmatization():
            field_data = self.__lemmatization_operation(field_data)
        if self.get_stemming():
            field_data = self.__stemming_operation(field_data)
        if self.get_named_entity_recognition():
            field_data = self.__named_entity_recognition_operation(field_data)
        return field_data
