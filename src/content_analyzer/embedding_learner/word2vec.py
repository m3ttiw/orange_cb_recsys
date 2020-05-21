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
               "source = " + str(self.__source) + \
               "preprocessor = " + str(self.__preprocessor) + " >"

    def __iter__(self):
        data_to_train = list()
        for line in self.__source:
            for field in self.__field_list:
                data_to_train.append(self.__preprocessor.process(line[field].lower()))

    def fit(self):
        """"
        Implementation of the Abstract Method start_training in the Abstract Class Word2vec.
        """
        model = Word2Vec(size=self.__size, window=self.__window, min_count=self.__min_count)
        total_examples = model.corpus_count
        model.build_vocab(senteces=GensimWord2Vec(self.__source, self.__preprocessor, self.__field_list))
        model.train(sentences=GensimWord2Vec(self.__source, self.__preprocessor, self.__field_list),
                    total_examples=total_examples, epochs=self.__epochs)
        model.save("word2vec.model")
        model_list = list()
        n = model.corpus_count
        for i in range(n):
            print("Doc-{}: {}".format(i, model.wv.__getitem__([str(i)])))
            model_list.append(model.wv.__getitem__([str(i)]))
        return model_list
