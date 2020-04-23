from src.offline.content_analyzer.field_content_production_technique import EmbeddingSource


class BinaryFile(EmbeddingSource):
    def __init__(self, file_path: str):
        super().__init__()
        self.__file_path: str = file_path

    def load(self):
        pass

# your embedding source
