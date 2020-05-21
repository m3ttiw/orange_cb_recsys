from src.content_analyzer.embedding_learner.embedding_learner import EmbeddingLearner
from src.content_analyzer.information_processor.information_processor import NLP
from src.content_analyzer.raw_information_source import RawInformationSource
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class GensimDoc2Vec(EmbeddingLearner):
    """"
    Class that implements the Abstract Class Word2Vec.
    Implementation of Word2Vec using the Gensim library.
    """
    def __init__(self, source: RawInformationSource,
                 preprocessor: NLP,
                 field_name: str,
                 **kwargs):
        super().__init__(source, preprocessor)
        self.__field_name = field_name

        if "max_epochs" in kwargs.keys():
            self.__max_epochs = kwargs["max_epochs"]
        else:
            self.__max_epochs = 100

        if "vec_size" in kwargs.keys():
            self.__vec_size = kwargs["vec_size"]
        else:
            self.__vec_size = 20

        if "alpha" in kwargs.keys():
            self.__alpha = kwargs["alpha"]
        else:
            self.__alpha = 0.025

    def __str__(self):
        return "GensimDoc2Vec"

    def __repr__(self):
        return "< GensimDoc2Vec :" + \
               "loader = " + str(self.__source) + \
               "preprocessor = " + str(self.__preprocessor) + " >"

    def start_learning(self):
        """"
        Implementation of the Abstract Method start_training in the Abstract Class Doc2vec.
        """

        data = list()
        # iter the source
        for doc in self.get_source():
            # apply preprocessing and save the data in the list
            data.append(self.get_preprocessor().process(doc[self.__field_name].lower()))

        #data = [self.get_preprocessor().process(field_data=doc[self.__field_name].lower()) for doc in self.get_source()]

        tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(data)]

        model = Doc2Vec(vector_size=self.__vec_size,
                        alpha=self.__alpha,
                        min_alpha=0.00025,
                        min_count=1,
                        dm=1)

        model.build_vocab(tagged_data)  # this create the vocabulary

        for epoch in range(self.__max_epochs):
            #print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            model.alpha -= 0.0002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        model.save("d2v.model")
        """
        return_list = list()
        n = model.docvecs.count
        for i in range(n):
            #print("Doc-{}: {}".format(i, model.docvecs[str(i)]))
            return_list.append(model.docvecs[str(i)])
        """
        return model

