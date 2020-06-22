import lzma
import pickle
from typing import Dict

from orange_cb_recsys.content_analyzer.field_content_production_techniques import BabelPyEntityLinking, LuceneTfIdf, \
    BinaryFile, GensimDownloader, Centroid, EmbeddingTechnique, SearchIndexing
from orange_cb_recsys.content_analyzer.field_content_production_techniques.synset_document_frequency import \
    SynsetDocumentFrequency
from orange_cb_recsys.content_analyzer.field_content_production_techniques.tf_idf import SkLearnTfIdf
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.lod_properties_retrieval import DBPediaMappingTechnique
from orange_cb_recsys.content_analyzer.memory_interfaces import IndexInterface
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile, CSVFile, SQLDatabase, DATFile
from orange_cb_recsys.utils.const import logger

""" Default dict for all implementation of the abstract classes, for different purpose, 
    with an 'alias' as key and the 'class name' as value
    You can use this to show all implemented class in the framework
    If a class is added to the framework and is a 'runnable_instance', 
    you must add to this dict using add_runnable_instance() function 
    or you can add manually in this dict and call __serialize() function 
    with no arguments to add it permanently and also show in this file
    """
runnable_instances = {
    "json": JSONFile,
    "csv": CSVFile,
    "sql": SQLDatabase,
    "dat": DATFile,
    "index": IndexInterface,
    "babelpy": BabelPyEntityLinking,
    "nltk": NLTK,
    "lucene_tf-idf": LuceneTfIdf,
    "binary_file": BinaryFile,
    "gensim_downloader": GensimDownloader,
    "centroid": Centroid,
    "embedding": EmbeddingTechnique,
    "text_blob_sentiment": TextBlobSentimentAnalysis,
    "number_normalizer": NumberNormalizer,
    "search_index": SearchIndexing,
    "sk_learn_tf-idf": SkLearnTfIdf,
    "dbpedia_mapping": DBPediaMappingTechnique,
    "synset_frequency": SynsetDocumentFrequency,
}


def __serialize(r_i=None):
    if r_i is None:
        r_i = runnable_instances
    logger.info("Serializing runnable_instances in utils dir",)

    path = 'runnableinstances.xz'
    with lzma.open(path, 'wb') as f:
        pickle.dump(r_i, f)


def get(alias: str = None):
    logger.info("Loading runnable_instances")
    r_i = {}
    try:
        path = 'runnableinstances.xz'
        with lzma.open(path, "rb") as f:
            r_i = pickle.load(f)
    except FileNotFoundError:
        logger.info('runnable_instances not found, dict is empty')
    if alias is None:
        return r_i
    elif alias in r_i.keys():
        return r_i[alias]
    else:
        logger.info('runnable_instance with %s alias not found', alias)
        return None


def add(alias: str, runnable_class: object):
    r_i = get()
    if alias in r_i.keys():
        logger.info('alias %s already exist, runnable_instance not added', alias)
    else:
        r_i[alias] = runnable_class
        __serialize(r_i)
        logger.info('%s successfully added', alias)


def remove(alias: str):
    r_i = get()
    if alias not in r_i.keys():
        logger.info('alias %s does not exist, runnable_instance not removed', alias)
    else:
        r_i.pop(alias)
        __serialize(r_i)
        logger.info('alias %s successfully removed', alias)


def show():
    r_i = get()
    for k in r_i.keys():
        logger.info('< %s : %s >', k, str(r_i[k]))


__serialize(runnable_instances)
