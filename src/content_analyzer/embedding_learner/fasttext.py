from src.content_analyzer.embedding_learner import embedding_learner
from src.content_analyzer.information_processor.information_processor import NLP
from gensim.models.fasttext import FastText
from src.content_analyzer.raw_information_source import RawInformationSource


class GensimFastText(embedding_learner.FastText):
    """"
    Class that implements the Abstract Class FastText.
    Implementation of FastText using the Gensim library.
    """

    def __init__(self, source: RawInformationSource,
                 preprocessor: NLP,
                 field_name: str,
                 **kwargs):
        preprocessor.set_is_tokenized(True)
        super().__init__(preprocessor)
        self.__field_name = field_name
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

        if "workers" in kwargs.keys():
            self.__workers = kwargs["workers"]

        if "alpha" in kwargs.keys():
            self.__alpha = kwargs["alpha"]

        if "min_alpha" in kwargs.keys():
            self.__min_alpha = kwargs["min_alpha"]

        if "sg" in kwargs.keys():
            self.__sg = kwargs["sg"]

        if "hs" in kwargs.keys():
            self.__hs = kwargs["hs"]

        if "seed" in kwargs.keys():
            self.__seed = kwargs["seed"]

        if "max_vocab_size" in kwargs.keys():
            self.__max_vocab_size = kwargs["max_vocab_size"]

        if "sample" in kwargs.keys():
            self.__sample = kwargs["sample"]

        if "negative" in kwargs.keys():
            self.__negative = kwargs["negative"]

        if "ns_exponent" in kwargs.keys():
            self.__ns_exponent = kwargs["ns_exponent"]

        if "cbow_mean" in kwargs.keys():
            self.__cbow_mean = kwargs["cbow_mean"]

        if "hashfxn" in kwargs.keys():
            self.__hashfxn = kwargs["hashfxn"]

        if "iter" in kwargs.keys():
            self.__iter = kwargs["iter"]

        if "trim_rule" in kwargs.keys():
            self.__trim_rule = kwargs["trim_rule"]

        if "batch_words" in kwargs.keys():
            self.__batch_words  = kwargs["batch_words"]

        if "min_n" in kwargs.keys():
            self.__min_n = kwargs["min_n"]

        if "max_n" in kwargs.keys():
            self.__max_n = kwargs["max_n"]

        if "word_ngrams" in kwargs.keys():
            self.__word_ngrams = kwargs["word_ngrams"]

        if "bucket" in kwargs.keys():
            self.__bucket = kwargs["bucket"]

        if "callbacks" in kwargs.keys():
            self.__callbacks = kwargs["callbacks"]

        if "compatible_hash" in kwargs.keys():
            self.__compatible_hash = kwargs["compatible_hash"]

        if "sorted_vocab" in kwargs.keys():
            self.__sorted_vocab = kwargs["sorted_vocab"]

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
            data_to_train.append(self.__preprocessor.process(line[self.__field_name].lower()))

    def start_learning(self):
        """"
        Implementation of the Abstract Method start_training in the Abstract Class FastText.
        """
        model = self.get_model()
        model.build_vocab(senteces=GensimFastText(self.__source, self.__preprocessor, self.__field_name))
        model_list = list()
        n = model.corpus_count
        for i in range(n):
            print("Doc-{}: {}".format(i, model.wv.__getitem__([str(i)])))
            model_list.append(model.wv.__getitem__([str(i)]))
        return model_list

    def get_model(self):
        """
        Method that returns a gensim FastText model
        """
        model = FastText(size=self.__size, window=self.__window, min_count=self.__min_count)
        total_examples = model.corpus_count
        model.train(sentences=GensimFastText(self.__source, self.__preprocessor, self.__field_name),
                    total_examples=total_examples, epochs=self.__epochs)
        return model

    def save_model(self):
        """
        Method used to save the curret model trained with FastText
        """
        model = self.get_model()
        model.save("fasttext.model")
