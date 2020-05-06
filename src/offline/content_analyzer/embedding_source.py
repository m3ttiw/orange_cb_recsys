from enum import Enum

import gensim.downloader as downloader
from gensim.models import KeyedVectors, Doc2Vec, fasttext
from src.offline.content_analyzer.field_content_production_technique import EmbeddingSource


class EmbeddingType(Enum):
    """
    Embeddings can be learned using different techniques,
    embeddings learned in different ways need to be loaded in different ways,
    so embedding_type needs to be specified.
    """
    WORD2VEC = 1
    DOC2VEC = 2
    FASTTEXT = 3


class BinaryFile(EmbeddingSource):
    """
    Class that implements the abstract class EmbeddingSource,
    this class loads the embeddings from a binary file
    in a way that depends from embedding type.

    Attributes:
        file_path (str): path for the binary file containing the embeddings
        embedding_type (EmbeddingType):
            technique used to learn the embedding that is being loaded
    """

    def __init__(self, file_path: str, embedding_type: EmbeddingType):
        super().__init__()
        self.__file_path: str = file_path
        if embedding_type == EmbeddingType.WORD2VEC:
            self.set_model(KeyedVectors.load_word2vec_format(self.__file_path, binary=True))
        elif embedding_type == EmbeddingType.DOC2VEC:
            self.set_model(Doc2Vec.load(self.__file_path))
        else:
            self.set_model(fasttext.load_facebook_vectors(self.__file_path))


class GensimDownloader(EmbeddingSource):
    """
    Class that implements the abstract class EmbeddingSource.
    This class loads the embeddings using the gensim downloader API.

    Attributes:
        name (str): name of the embeddings model to be loaded
    """

    def __init__(self, name: str):
        super().__init__()
        self.__name: str = name
        self.set_model(downloader.load(self.__name))


# your embedding source
