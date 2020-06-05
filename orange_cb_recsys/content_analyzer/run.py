import json
import sys
import yaml
from typing import List, Dict

import lucene

from orange_cb_recsys.content_analyzer.config import ContentAnalyzerConfig, FieldConfig, \
    FieldRepresentationPipeline
from orange_cb_recsys.content_analyzer.content_analyzer_main import ContentAnalyzer
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    Centroid
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.embedding_source import \
    BinaryFile, GensimDownloader
from orange_cb_recsys.content_analyzer.field_content_production_techniques.entity_linking import BabelPyEntityLinking
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    EmbeddingTechnique, SearchIndexing
from orange_cb_recsys.content_analyzer.field_content_production_techniques.tf_idf import LuceneTfIdf
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.memory_interfaces.text_interface import IndexInterface
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsImporter, RatingsFieldConfig
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile, SQLDatabase, CSVFile

lucene.initVM(vmargs=['-Djava.awt.headless=true'])

DEFAULT_CONFIG_PATH = "config.json"

implemented_preprocessing = [
    "nltk",
]

implemented_content_prod = [
    "embedding",
    "babelpy",
    "lucene_tf-idf",
    "search_index"
]

implemented_rating_proc = [
    "text_blob_sentiment",
    "number_normalizer",
]

runnable_instances = {
    "json": JSONFile,
    "csv": CSVFile,
    "sql": SQLDatabase,
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
    "search_index": SearchIndexing
}


def check_for_available(content_config: Dict):
    # check if need_interface is respected
    # check runnable_instances
    if content_config['source_type'] not in ['json', 'csv', 'sql']:
        return False
    if content_config['content_type'].lower() == 'ratings':
        if "from" not in content_config.keys() \
                or "to" not in content_config.keys() \
                or "timestamp" not in content_config.keys()\
                or "output_directory" not in content_config.keys():
            return False
        for field in content_config['fields']:
            if field['rating_processor']['class'] not in implemented_rating_proc:

                return False
        return True
    for field_dict in content_config['fields']:
        if field_dict['memory_interface'] not in ['index', 'None']:
            return False
        for pipeline_dict in field_dict['pipeline_list']:
            if pipeline_dict['field_content_production'] != "None":
                if pipeline_dict['field_content_production']['class'] not in implemented_content_prod:
                    return False
            for preprocessing in pipeline_dict['preprocessing_list']:
                if preprocessing['class'] not in implemented_preprocessing:
                    return False
    return True


def dict_detector(technique_dict):
    """
    detect a a class constructor call in a sub-dict of a dict
    """
    for key in technique_dict.keys():
        value = technique_dict[key]
        if type(value) == dict and 'class' in value.keys():
            parameter_class_name = value.pop('class')
            technique_dict[key] = runnable_instances[parameter_class_name](**value)

    return technique_dict


def content_config_run(config_list: List[Dict]):
    for content_config in config_list:
        # content production
        content_analyzer_config = ContentAnalyzerConfig(
            content_config["content_type"],
            runnable_instances[content_config['source_type']](file_path=config_dict["raw_source_path"]),
            content_config['id_field_name'],
            content_config['output_directory'],
            content_config['search_index'])

        for field_dict in content_config['fields']:
            field_config = FieldConfig()
            # setting the content analyzer config

            for pipeline_dict in field_dict['pipeline_list']:
                preprocessing_list = list()
                for preprocessing in pipeline_dict['preprocessing_list']:
                    # each preprocessing settings
                    class_name = preprocessing.pop('class')  # extract the class acronyms
                    preprocessing = dict_detector(preprocessing)
                    preprocessing_list.append(runnable_instances[class_name](**preprocessing))  # params for the class
                # content production settings
                if type(pipeline_dict['field_content_production']) is dict:
                    class_name = pipeline_dict['field_content_production'].pop('class')  # extract the class acronyms
                    # append each field representation pipeline to the field config
                    technique_dict = pipeline_dict["field_content_production"]
                    technique_dict = dict_detector(technique_dict)
                    field_config.append_pipeline(
                        FieldRepresentationPipeline(runnable_instances[class_name](**technique_dict),
                                                    preprocessing_list))
                else:
                    field_config.append_pipeline(FieldRepresentationPipeline(None, preprocessing_list))
            # verify that the memory interface is set
            if field_dict['memory_interface'] != "None":
                field_config.set_memory_interface(runnable_instances[field_dict['memory_interface']](
                    field_dict['memory_interface_path']))
            content_analyzer_config.append_field_config(field_dict["field_name"], field_config)

        # fitting the data for each
        content_analyzer = ContentAnalyzer(content_analyzer_config)  # need the id list (id configuration)
        content_analyzer.fit()


def rating_config_run(config_dict: Dict):
    rating_configs = []
    for field in config_dict["fields"]:
        rating_configs.append(
            RatingsFieldConfig(preference_field_name=field["preference_field_name"],
                               processor=dict_detector(field["processor"]))
        )
    RatingsImporter(
        source=runnable_instances[config_dict["source_type"]](file_path=config_dict["raw_source_path"], **config_dict),
        output_directory=config_dict["output_directory"],
        rating_configs=rating_configs,
        from_field_name=config_dict["from_field_name"],
        to_field_name=config_dict["to_field_name"],
        timestamp_field_name=config_dict["timestamp"]
    ).import_ratings()


if __name__ == "__main__":
    config_path = sys.argv[0]
    if config_path is not None:
        config_path = DEFAULT_CONFIG_PATH
    if config_path.endswith('.yml'):
        config_list_dict = yaml.load(open(config_path), Loader=yaml.FullLoader)
    elif config_path.endswith('.json'):
        config_list_dict = json.load(open(config_path))
    else:
        raise Exception("Wrong file extension")

    for config_dict in config_list_dict:
        if check_for_available(config_dict):
            if config_dict["content_type"].lower() == "rating":
                rating_config_run(config_dict)
            else:
                content_config_run([config_dict])
        else:
            raise Exception("Check for available instances failed.")
