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
        super().__init__(source, preprocessor, field_list)
        self.optionals = {}
        if "corpus_file" in kwargs.keys():
            self.optionals["corpus_file"] = kwargs["corpus_file"]

        if "size" in kwargs.keys():
            self.optionals["size"] = kwargs["size"]

        if "window" in kwargs.keys():
            self.optionals["window"] = kwargs["window"]

        if "min_count" in kwargs.keys():
            self.optionals["min_count"] = kwargs["min_count"]

        if "workers" in kwargs.keys():
            self.optionals["workers"] = kwargs["workers"]

        if "alpha" in kwargs.keys():
            self.optionals["alpha"] = kwargs["alpha"]

        if "min_alpha" in kwargs.keys():
            self.optionals["min_alpha"] = kwargs["min_alpha"]

        if "sg" in kwargs.keys():
            self.optionals["sg"] = kwargs["sg"]

        if "hs" in kwargs.keys():
            self.optionals["hs"] = kwargs["hs"]

        if "seed" in kwargs.keys():
            self.optionals["seed"] = kwargs["seed"]

        if "max_vocab_size" in kwargs.keys():
            self.optionals["max_vocab_size"] = kwargs["max_vocab_size"]

        if "sample" in kwargs.keys():
            self.optionals["sample"] = kwargs["sample"]

        if "negative" in kwargs.keys():
            self.optionals["negative"] = kwargs["negative"]

        if "ns_exponent" in kwargs.keys():
            self.optionals["ns_exponent"] = kwargs["ns_exponent"]

        if "cbow_mean" in kwargs.keys():
            self.optionals["cbow_mean"] = kwargs["cbow_mean"]

        if "hashfxn" in kwargs.keys():
            self.optionals["hashfxn"] = kwargs["hashfxn"]

        if "iter" in kwargs.keys():
            self.optionals["iter"] = kwargs["iter"]

        if "trim_rule" in kwargs.keys():
            self.optionals["trim_rule"] = kwargs["trim_rule"]

        if "batch_words" in kwargs.keys():
            self.optionals["batch_words"] = kwargs["batch_words"]

        if "min_n" in kwargs.keys():
            self.optionals["min_n"] = kwargs["min_n"]

        if "max_n" in kwargs.keys():
            self.optionals["max_n"] = kwargs["max_n"]

        if "word_ngrams" in kwargs.keys():
            self.optionals["word_ngrams"] = kwargs["word_ngrams"]

        if "bucket" in kwargs.keys():
            self.optionals["bucket"] = kwargs["bucket"]

        if "callbacks" in kwargs.keys():
            self.optionals["callbacks"] = kwargs["callbacks"]

        if "compatible_hash" in kwargs.keys():
            self.optionals["compatible_hash"] = kwargs["compatible_hash"]

        if "sorted_vocab" in kwargs.keys():
            self.optionals["sorted_vocab"] = kwargs["sorted_vocab"]

        if "ephocs" in kwargs.keys():
            self.__epochs = kwargs["ephocs"]
        else:
            self.__epochs = 50

    def __str__(self):
        return "FastText"

    def __repr__(self):
        return "< GensimDoc2Vec :" + \
               "source = " + str(self.get_source()) + \
               "preprocessor = " + str(self.get_preprocessor()) + " >"

    def fit(self):
        """"
        Implementation of the Abstract Method fit in the Abstract Class EmbeddingLearner.

        Returns:
            generator: the model, trained on the data in the field_list variable, is returned
        """
        data_to_train = list()
        for line in self.get_source():
            doc = []
            for field_name in self.get_field_list():
                field_data = self.get_preprocessor().process(line[field_name].lower())
                doc.append(field_data)
            data_to_train.append(doc)
        model = FastText(sentences=data_to_train, **self.optionals)
        # model.build_vocab(senteces=GensimFastText(self.get_source(), self.get_preprocessor(), self.get_field_list()))
        model.train(data_to_train,
                    total_examples=model.corpus_count,
                    epochs=self.__epochs)
        return model

    def save(self):
        model = GensimFastText(source=self.get_source(),
                               preprocessor=self.get_preprocessor(),
                               field_list=self.get_field_list()).fit()
        model.save("fasttext.model")
