from typing import List
import time

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
                field_data = self.get_preprocessor().process(content[field_name])
                if type(field_data) is list:
                    field_data = ' '.join(field_data)
                document.append(field_data)
            corpus.append(document)

        dictionary = Dictionary(corpus)

        model = RpModel(corpus, id2word=dictionary)

        return model

    def save(self, model):
        model.save('random_indexing' + time.time())

