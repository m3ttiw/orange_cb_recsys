from typing import List

from src.content_analyzer.embedding_learner import embedding_learner
from src.content_analyzer.information_processor.information_processor import NLP
from gensim.models.fasttext import FastText
from src.content_analyzer.raw_information_source import RawInformationSource


class GensimFastText(embedding_learner.EmbeddingLearner):
    """"
    Class that implements the Abstract Class EmdeddingLearner.
    Implementation of FastText using the Gensim library.
    """

    def __init__(self, source: RawInformationSource,
                 preprocessor: NLP,
                 field_list: List[str],
                 **kwargs):
        super().__init__(preprocessor)
        self.__field_list = field_list
        self.__source = source

        if "size" in kwargs.keys():
            self.__size = kwargs["size"]
        else:
            self.__size = 4

        if "window" in kwargs.keys():
            self.__window = kwargs["window"]
        else:
            self.__window = 3

        if "min_count" in kwargs.keys():
            self.__min_count = kwargs["min_count"]
        else:
            self.__min_count = 1

        if "ephocs" in kwargs.keys():
            self.__epochs = kwargs["ephocs"]
        else:
            self.__epochs = 5

    def __str__(self):
        return "FastText"

    def __repr__(self):
        return "< FastText :" + \
               "preprocessor = " + str(self.__preprocessor) + " >"

    def __iter__(self):
        data_to_train = list()
        for line in self.__source:
            for field in self.__field_list:
                data_to_train.append(self.__preprocessor.process(line[field].lower()))

    def fit(self):
        """"
        Implementation of the Abstract Method fit in the Abstract Class EmbeddingLearner.
        """
        model = FastText(size=self.__size, window=self.__window, min_count=self.__min_count)
        total_examples = model.corpus_count
        model.build_vocab(senteces=GensimFastText(self.__source, self.__preprocessor, self.__field_name))
        model.train(sentences=GensimFastText(self.__source, self.__preprocessor, self.__field_name),
                    total_examples=total_examples, epochs=self.__epochs)
        model.save("fasttext.model")
        model_list = list()
        n = model.corpus_count
        for i in range(n):
            print("Doc-{}: {}".format(i, model.wv.__getitem__([str(i)])))
            model_list.append(model.wv.__getitem__([str(i)]))
        return model_list
