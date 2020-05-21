from enum import Enum
from typing import List

import gensim.downloader as downloader
from gensim.models import KeyedVectors, Doc2Vec, fasttext
from wikipedia2vec import Wikipedia2Vec
import numpy as np

from src.content_analyzer.field_content_production_techniques.field_content_production_technique import EmbeddingSource


class EmbeddingType(Enum):
    """
    Embeddings can be learned using different techniques.
    Embeddings learned in different ways need to be loaded in different ways,
    so embedding_type needs to be specified.
    """
    WORD2VEC = 1
    DOC2VEC = 2
    FASTTEXT = 3


class BinaryFile(EmbeddingSource):
    """
    Class that implements the abstract class EmbeddingSource.
    This class loads the embeddings from a binary file
    in a way that depends from embedding_type.

    Attributes:
        file_path (str): path for the binary file containing the embeddings
        embedding_type (EmbeddingType):
            technique used to learn the embedding that is being loaded
    """

    def __init__(self, file_path: str,
                 embedding_type: EmbeddingType):
        super().__init__()
        self.__file_path: str = file_path
        if embedding_type == 1:
            self.set_model(KeyedVectors.load_word2vec_format(self.__file_path, binary=True))
        elif embedding_type == 2:
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


class Wikipedia2VecDownloader(EmbeddingSource):
    """
    Class that implements the abstract class EmbeddingSoruce.
    This class loads the embeddings using the Wikipedia2Vec binary file loader.
    Can be used for loading of pre-trained wikipedia dump embedding, 
    both downloaded or trained on local machine.

    Attributes:
        path (str): path for the binary file containing the embeddings
    """

    def __init__(self, path: str):
        super().__init__()
        self.__path: str = path

        self.set_model(Wikipedia2Vec.load(self.__path))

    def get_vector_size(self) -> int:
        return self.get_model().get_word_vector("a").shape[0]

    def load(self, text: List[str]) -> np.ndarray:
        """
        Function that extracts from the embeddings model
        the vectors of the words contained in text

        Args:
            text (str): contains words of which vectors will be extracted

        Returns:
            np.ndarray: bi-dimensional numpy vector,
                each row is a term vector
        """
        embedding_matrix = np.ndarray(shape=(len(text), self.get_vector_size()))

        for i, word in enumerate(text):
            try:
                embedding_matrix[i, :] = self.get_model().get_word_vector(word)
            except:
                pass

        return embedding_matrix

# your embedding source
