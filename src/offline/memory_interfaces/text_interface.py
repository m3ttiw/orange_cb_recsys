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
        super().__init__(directory)

        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        fs_directory = SimpleFSDirectory(Paths.get(self.__directory))
        self.__writer = IndexWriter(fs_directory, IndexWriterConfig())

    def new_item(self):
        self.__doc = Document()

    def serialize(self, field_name: str, field_data):
        self.__doc.add(Field(field_name, field_data, StringField.TYPE_STORED))

    def close_item(self):
        self.__writer.addDocument(self.__doc)

    def __iter__(self):
        fs_dir = SimpleFSDirectory(Paths.get(self.__directory))
        searcher = IndexSearcher(DirectoryReader.open(fs_dir))

        for i in range(0, searcher.maxDoc()):
            doc = searcher.doc(i)
            item = {}
            for field in doc.iterator():
                item[field.name()] = field.stringValue()

            yield item
