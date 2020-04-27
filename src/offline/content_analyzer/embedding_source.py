from src.offline.content_analyzer.field_content_production_technique import EmbeddingSource


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

    def load(self):
        """
        Function that loads the embeddings from the file.

        Returns:
            The loaded embedding
        """
        print("loading pre-trained embedding stored in this binary file")

# your embedding source
