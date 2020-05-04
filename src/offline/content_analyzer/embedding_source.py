from src.offline.content_analyzer.field_content_production_technique import EmbeddingSource
import gensim.downloader as downloader
from gensim.models import KeyedVectors
import numpy as np


class BinaryFile(EmbeddingSource):
    """
    Class that implements the abstract class EmbeddingSource.
    This class loads the embeddings from a binary file.

    Attributes:
        file_path (str): Path for the binary file containing the embeddings
    """
    def __init__(self, file_path: str):
        super().__init__()
        self.__file_path: str = file_path
        self.__model = KeyedVectors.load_word2vec_format(self.__file_path, binary=True)

    def load(self, text: str):
        """
        Function that loads the embeddings from the file.

        Returns:
            The loaded embedding matrix
        """
        words = text.split(" ")
        embedding_matrix = np.ndarray(shape=(len(words), self.__model[words[0]].shape[0]))
        print(embedding_matrix.shape)

        for i, word in enumerate(words):
            embedding_matrix[i, :] = self.__model[word]

        return embedding_matrix


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
        self.__model = downloader.load(self.__name)

    def load(self, text: str):
        """
        Function that loads the embeddings downloading it using the gensim downloader api.

        Returns:
            The loaded embedding matrix
        """

        words = text.split(" ")
        embedding_matrix = np.ndarray(shape=(len(words), self.__model[words[0]].shape[0]))

        for i, word in enumerate(words):
            embedding_matrix[i, :] = self.__model[word]


        return embedding_matrix

# your embedding source
