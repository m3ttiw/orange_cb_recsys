from gensim.models import RpModel
from gensim.corpora import Dictionary

from src.content_analyzer.embedding_learner.embedding_learner import EmbeddingLearner


class RandomIndexing(EmbeddingLearner):
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
