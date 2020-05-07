import lucene
import math

from java.nio.file import Paths
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.document import Document, Field, StringField, TextField, FieldType
from org.apache.lucene.store import SimpleFSDirectory
from offline.memory_interfaces.memory_interfaces import TextInterface
from org.apache.lucene.util import BytesRefIterator
from org.apache.lucene.index import DirectoryReader, Term


class IndexInterface(TextInterface):
    """
    Abstract class that takes care of serializing and deserializing text in an indexed structure
    """
    def __init__(self, directory: str):
        super().__init__(directory)
        lucene.initVM(vmargs=['-Djava.awt.headless=true']) # controllare che non venga rieseguita
        self.__doc = None
        self.__writer = None
        self.__field_type = None

    def init_writing(self):
        self.__field_type = FieldType(StringField.TYPE_STORED)
        self.__field_type.setStored(True)
        self.__field_type.setTokenized(True)
        self.__field_type.setStoreTermVectors(True)
        self.__field_type.setStoreTermVectorPositions(True)
        self.__field_type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS)
        fs_directory = SimpleFSDirectory(Paths.get(self.get_directory()))
        self.__writer = IndexWriter(fs_directory, IndexWriterConfig())

    def new_content(self):
        """
        In the lucene index case the new content
        is a new document in the index
        Returns:

        """
        self.__doc = Document()

    def new_field(self, field_name: str, field_data):
        self.__doc.add(Field(field_name, field_data, StringField.TYPE_STORED))

    def serialize_item(self):
        self.__writer.addDocument(self.__doc)

    def stop_writing(self):
        self.__writer.commit()
        self.__writer.close()

    def get_tf_idf(self, field_data, field_name: str, item_id: str):
        reader = DirectoryReader.open(SimpleFSDirectory(Paths.get(self.get_directory())))
        term_vector = reader.getTermVector(0, field_name)
        term_enum = term_vector.iterator()
        words_bag = {}
        for term in BytesRefIterator.cast_(term_enum):
            postings = term_enum.postings(None)
            postings.nextDoc()
            term_text = term.utf8ToString()
            term_frequency = 1 + math.log(postings.freq())  # normalized term frequency
            inverse_document_frequency = math.log10(reader.maxDoc() / reader.docFreq(Term(field_name, term)))
            tf_idf = term_frequency * inverse_document_frequency
            words_bag[term_text] = tf_idf

        return words_bag
