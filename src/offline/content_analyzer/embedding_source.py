from typing import List

from offline.content_analyzer.field_content_production_technique import EmbeddingSource
import gensim.downloader as downloader
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
        self.__model = None

    def load(self, text: str):
        """
        Function that loads the embeddings from the file.

        Returns:
            The loaded embedding
        """

        words = text.split(" ")

        embedding_matrix = np.ndarray(shape=(len(text, )))

        for i, word in enumerate(words):
            embedding_matrix[i, :] = self.__model[word]


class GensimDownloader(EmbeddingSource):
    """
    Class that implements the abstract class EmbeddingSource.
    This class loads the embeddings from a binary file.

    Attributes:
        model_name:
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.__model_name = model_name
        self.__model = downloader.load(self.__model_name).wv

    def load(self, text: str):
        """
        Function that loads the embeddings from the file.

        Returns:
            Embedding matrix, one row for each vector in text
        """

        words = text.split(" ")

        embedding_matrix = np.ndarray(shape=(len(text, )))

        for i, word in enumerate(words):
            embedding_matrix[i, :] = self.__model[word]

# your embedding source
