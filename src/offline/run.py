import json
import sys
from typing import List, Dict

from src.offline.content_analyzer.content_analyzer_main import ContentAnalyzer, FieldConfig, ContentAnalyzerConfig, \
    FieldRepresentationPipeline
from src.offline.content_analyzer.entity_linking import BabelPyEntityLinking
from src.offline.content_analyzer.field_content_production_technique import TfIdfTechnique
from src.offline.content_analyzer.nlp import OpenNLP
from src.offline.memory_interfaces.text_interface import IndexInterface
from src.offline.raw_data_extractor.raw_data_manager import RawDataConfig, RawDataManager
from src.offline.raw_data_extractor.raw_information_source import JSONFile, CSVFile, SQLDatabase

DEFAULT_CONFIG_PATH = ".\config.json"

implemented_preprocessing = [
    "open_nlp",
]

implemented_content_prod = [
    "embedding",
    "babelpy",
    "tf-idf",
]

runnable_instances = {
    "json": JSONFile,
    "csv": CSVFile,
    "sql": SQLDatabase,
    "index": IndexInterface,
    "babelpy": BabelPyEntityLinking,
    "open_nlp": OpenNLP,
    "tf-idf": TfIdfTechnique,
}

need_interface = [
    "tf-idf",
]


def check_for_available(config_list: List[Dict]):
    # check if need_interface is respected
    # check runnable_instances
    check_pass = True
    for content_config in config_list:
        if content_config['source_type'] not in ['json', 'csv', 'sql']:
            check_pass = False
            break
        for field_dict in content_config['fields']:
            if field_dict['memory_interface'] not in ['index', 'None']:
                check_pass = False
                break
            for pipeline_dict in field_dict['pipeline_list']:
                if pipeline_dict['field_content_production']['class'] not in implemented_content_prod:
                    check_pass = False
                    break
                for preprocessing in pipeline_dict['preprocesing_list']:
                    if preprocessing['class'] not in implemented_preprocessing:
                        check_pass = False
                        break
    return check_pass


def config_run(config_list: List[Dict]):
    representend_content_list = list()
    for content_config in config_list:

        # phase 1 : memorize the selected fields with some high powered memory interface
        config_dict = {}
        for field_dict in content_config['fields']:

            # verify that the memory interface is set
            if field_dict['memory_interface'] != "None":
                # setting the config dict for each
                config_dict[field_dict['field_name']] = runnable_instances[field_dict['memory_interface']](
                    field_dict['memory_interface_path'])

        # setting the phase 1 configuration
        raw_data_config = RawDataConfig(
            runnable_instances[content_config['source_type']](content_config['raw_source_path']),
            content_config['id_field_name'], config_dict)
        RawDataManager(raw_data_config).fit()

        # phase 2 : content production
        field_config = FieldConfig()
        for field_dict in content_config['fields']:
            # setting the content analyzer config
            content_analyzer_config = ContentAnalyzerConfig(
                runnable_instances[content_config['source_type']](content_config['raw_source_path']),
                content_config['id_field_name'])
            for pipeline_dict in field_dict['pipeline_list']:
                preprocessing_list = list()
                for preprocessing in pipeline_dict['preprocesing_list']:
                    # each preprocessing settings
                    class_name = preprocessing.pop('class')  # extract the class acronyms
                    preprocessing_list.append(runnable_instances[class_name](**preprocessing))  # params for the class
                # content production settings
                class_name = pipeline_dict['field_content_production'].pop('class')  # extract the class acronyms
                if class_name in need_interface:
                    pipeline_dict['memory_interface'] = content_config['memory_interface']
                # append each field representation pipeline to the field config
                field_config.add_pipeline(FieldRepresentationPipeline(runnable_instances[class_name](**pipeline_dict),
                                                                      preprocessing_list))

        # fitting the data for each
        content_analyzer = ContentAnalyzer(content_analyzer_config)  # need the id list (id configuration)
        representend_content_list.append(content_analyzer.fit())

    return representend_content_list


if __name__ == "__main__":
    config_path = sys.argv[0]
    if config_path is not None:
        config_path = DEFAULT_CONFIG_PATH
    config_list_dict = json.load(open(config_path))
    if check_for_available(config_list_dict):
        config_run(config_list_dict)
    else:
        raise Exception("Check for available instances failed.")
