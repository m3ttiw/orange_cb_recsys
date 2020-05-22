from typing import List

from gensim.models import Word2Vec

from src.content_analyzer.embedding_learner import embedding_learner
from src.content_analyzer.information_processor.information_processor import InformationProcessor
from src.content_analyzer.raw_information_source import RawInformationSource


class GensimWord2Vec(embedding_learner.EmbeddingLearner):
    """"
    Class that implements the Abstract Class Word2Vec.
    Implementation of Word2Vec using the Gensim library.
    """

    def __init__(self, source: RawInformationSource,
                 preprocessor: InformationProcessor,
                 field_list=List[str],
                 **kwargs):
        super().__init__(source, preprocessor, field_list)

        if "size" in kwargs.keys():
            self.__size = kwargs["size"]
        else:
            self.__size = 100

        if "window" in kwargs.keys():
            self.__window = kwargs["window"]
        else:
            self.__window = 5

        if "min_count" in kwargs.keys():
            self.__min_count = kwargs["min_count"]
        else:
            self.__min_count = 1

        if "ephocs" in kwargs.keys():
            self.__epochs = kwargs["ephocs"]
        else:
            self.__epochs = 50

    def __str__(self):
        return "GensimWord2Vec"

    def __repr__(self):
        return "< GensimWord2Vec :" + \
               "source = " + str(self.get_source()) + \
               "preprocessor = " + str(self.get_preprocessor()) + " >"

    def fit(self):
        """"
        Implementation of the Abstract Method start_training in the Abstract Class Word2vec.
        """
        data_to_train = list()
        for line in self.get_source():
            doc = []
            for field_name in self.get_field_list():
                field_data = self.get_preprocessor().process(line[field_name].lower())
                if type(field_data) is list:
                    field_data = ' '.join(field_data)
                doc.append(field_data)
            data_to_train.append(doc)
        model = Word2Vec(sentences=data_to_train)
        model.build_vocab(data_to_train)
        model.train(sentences=data_to_train,
                    total_examples=model.corpus_count,
                    epochs=self.__epochs)
        return model

    def save(self):
        model = GensimWord2Vec(source=self.get_source(),
                               preprocessor=self.get_preprocessor(),
                               field_list=self.get_field_list()).fit()
        model.save("word2vec.model")
