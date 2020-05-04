from abc import ABC

from offline.content_analyzer.information_processor import NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class NLTK(NLP):
    """
    Interface for the library Opennlp for natural language processing features

    """
    def __init__(self, stopwords_removal: bool = False,
                 stemming: bool = False,
                 lemmatization: bool = False,
                 named_entity_recognition: bool = False,
                 strip_multiple_whitespaces: bool = True,
                 url_tagging: bool = False,
                 lan: str = "english"
                 ):

        super().__init__(stopwords_removal,
                         stemming, lemmatization, named_entity_recognition,
                         strip_multiple_whitespaces, url_tagging)
        self.__lan = lan
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download()
    """
    def get_sp(self):
        return self.__sp

    def set_sp(self, lan):
        self.__sp = spacy.load(lan)
    """
    def get_lan(self):
        return self.__lan

    def set_lan(self, lan):
        self.__lan = lan

    def __stopwords_removal(self, text):
        stop_words = set(stopwords.words(self.get_lan()))
        word_tokens = word_tokenize(text)
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)

        return ' '.join(filtered_sentence)

    def __stemming(self, text):
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer(language=self.get_lan())
        splitted_text = text.split()
        stemmed_text = []
        for word in splitted_text:
            stemmed_text.append(stemmer.stem(word))

        return ' '.join(stemmed_text)

    def __get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def __lemmatization(self, text):
        lemmatizer = WordNetLemmatizer()
        splitted_text = text.split()
        lemmatized_text = []
        for word in splitted_text:
            lemmatized_text.append(lemmatizer.lemmatize(word, self.__get_wordnet_pos(word)))
        return ' '.join(lemmatized_text)

    def __named_entity_recognition(self, text):
        text = nltk.word_tokenize(text)
        text = nltk.pos_tag(text)
        pattern = 'NP: {<DT>?<JJ>*<NN>}'
        cp = nltk.RegexpParser(pattern)
        cs = cp.parse(text)
        return cs

    def __strip_multiple_whitespaces(self, text):
        import re
        return re.sub(' +', ' ', text)

    def __url_tagging(self, text):
        import re
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]| [!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                          text)
        for url in urls:
            text = text.replace(url, "<URL>")
        return text

    def process(self, field_data):
        if self.get_url_tagging():
            field_data = self.__url_tagging(field_data)
        if self.get_strip_multiple_whitespaces():
            field_data = self.__strip_multiple_whitespaces(field_data)
        if self.get_stopwords_removal():
            field_data = self.__stopwords_removal(field_data)
        if self.get_lemmatization():
            field_data = self.__lemmatization(field_data)
        if self.get_stemming():
            field_data = self.__stemming(field_data)
        if self.get_named_entity_recognition():
            field_data = self.__named_entity_recognition(field_data)
        return field_data