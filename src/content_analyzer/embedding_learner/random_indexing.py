from typing import List
import time

from gensim.models import RpModel
from gensim.corpora import Dictionary

from src.content_analyzer.embedding_learner.embedding_learner import EmbeddingLearner
from src.content_analyzer.information_processor.information_processor import TextProcessor
from src.content_analyzer.raw_information_source import RawInformationSource


class RandomIndexing(EmbeddingLearner):
    def __init__(self, source: RawInformationSource,
                 preprocessor: TextProcessor,
                 field_list: List[str]):
        super().__init__(source, preprocessor, field_list)

    def fit(self):
        corpus = self.extract_corpus()

        dictionary = Dictionary(corpus)

        model = RpModel(corpus, id2word=dictionary)

        return model
