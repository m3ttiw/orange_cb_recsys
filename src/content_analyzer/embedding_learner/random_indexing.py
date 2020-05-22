from typing import List

from gensim.models import RpModel
from gensim.corpora import Dictionary

from src.content_analyzer.embedding_learner.embedding_learner import EmbeddingLearner
from src.content_analyzer.information_processor.information_processor import InformationProcessor
from src.content_analyzer.raw_information_source import RawInformationSource


class RandomIndexing(EmbeddingLearner):
    def __init__(self, source: RawInformationSource,
                 preprocessor: InformationProcessor,
                 field_list: List[str]):
        super().__init__(source, preprocessor, field_list)

    def fit(self):
        corpus = []

        for content in self.get_source():
            document = []
            for field_name in self.get_field_list():
                document.append(content[field_name])
            corpus.append(document)

        dictionary = Dictionary(corpus)

        model = RpModel(corpus, id2word=dictionary)

        return model
