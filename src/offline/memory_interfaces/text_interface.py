import lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.store import SimpleFSDirectory
from offline.memory_interfaces.memory_interfaces import TextInterface
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import DirectoryReader


class IndexInterface(TextInterface):
    """
    Abstract class that takes care of serializing and deserializing text in an indexed structure
    """
    def __init__(self, directory: str):
        self.__doc = None
        self.__writer = None
        super().__init__(directory)

    def init_writing(self):
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        fs_directory = SimpleFSDirectory(Paths.get(self.get_directory()))
        self.__writer = IndexWriter(fs_directory, IndexWriterConfig())

    def new_item(self):
        self.__doc = Document()

    def new_field(self, field_name: str, field_data):
        self.__doc.add(Field(field_name, field_data, StringField.TYPE_STORED))

    def serialize_item(self):
        self.__writer.addDocument(self.__doc)

    def stop_writing(self):
        self.__writer.commit()
        self.__writer.close()
