from src.content_analyzer.embedding_learner import embedding_learner
from src.content_analyzer.information_processor.information_processor import InformationProcessor
from src.content_analyzer.raw_information_source import RawInformationSource
from gensim.test.utils import get_tmpfile
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class GensimDoc2Vec(embedding_learner.Doc2Vec):
    """"
    Class that implements the Abstract Class Word2Vec.
    Implementation of Word2Vec using the Gensim library.
    """
    is_first_instance = True

    def __init__(self, source: RawInformationSource,
                 preprocessor: InformationProcessor,
                 **kwargs):
        super().__init__(source, preprocessor)

        if "model_path" in kwargs.keys():
            self.__fname = get_tmpfile(str(kwargs["model_path"]))
        else:
            self.__fname = get_tmpfile("doc2vec_model")

        if GensimDoc2Vec.is_first_instance:
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
            self.__model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
        elif "use_pre_trained_model" in kwargs.keys():
            self.__model = Doc2Vec.load(self.__fname)  # you can continue training with the loaded model!

        self.__model.save(self.__fname)
        GensimDoc2Vec.is_first_instance = False

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

        print("learning")
