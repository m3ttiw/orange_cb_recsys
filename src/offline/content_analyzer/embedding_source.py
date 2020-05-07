from enum import Enum

from src.offline.content_analyzer.field_content_production_technique import EmbeddingSource
import gensim.downloader as downloader
from gensim.models import KeyedVectors, Doc2Vec, fasttext


class EmbeddingType(Enum):
    WORD2VEC = 1
    DOC2VEC = 2
    FASTTEXT = 3


class BinaryFile(EmbeddingSource):
    """
    Class that implements the abstract class EmbeddingSource.
    This class loads the embeddings from a binary file.

    Attributes:
        file_path (str): Path for the binary file containing the embeddings
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
    This class loads the embeddings from a binary file.

    Attributes:
        name (str): Path for the binary file containing the embeddings
    """

    def __init__(self, name: str):
        super().__init__()
        self.__name: str = name
        self.set_model(downloader.load(self.__name))


# your embedding source
