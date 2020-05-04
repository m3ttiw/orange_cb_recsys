import json

from src.offline.content_analyzer.content_analyzer_main import ContentAnalyzer, FieldConfig, ContentAnalyzerConfig, \
    FieldRepresentationPipeline
from src.offline.content_analyzer.entity_linking import BabelPyEntityLinking
from src.offline.content_analyzer.nlp import OpenNLP
from src.offline.memory_interfaces.text_interface import IndexInterface
from src.offline.movielens_example import content_analyzer_config
from src.offline.raw_data_extractor.raw_data_manager import RawDataConfig, RawDataManager
from src.offline.raw_data_extractor.raw_information_source import JSONFile, CSVFile, SQLDatabase

runnable_istances = {
    "json": JSONFile,
    "csv": CSVFile,
    "sql": SQLDatabase,
    "index": IndexInterface,
    "babelpy": BabelPyEntityLinking,
    "open_nlp": OpenNLP,

}

def config_run(config_path: str = ".\config.json"):
    config_list = json.load(open(config_path))
    for content_config in config_list:
        print(content_config['content_type'])
        print(content_config['raw_source_path'])
        print(content_config['id_field_name'])

        # fase 1
        for field_dict in content_config['fields']:
            if field_dict['memory_interface'] != "None":
                config_dict = {field_dict['field_name']: runnable_istances[field_dict['memory_interface']]('./test-index-plot')}
                raw_data_config = RawDataConfig(
                    runnable_istances[content_config['source_type']](content_config['raw_source_path']),
                    content_config['id_field_name'], config_dict)
                RawDataManager(raw_data_config).fit()

        # fase 2
        for field_dict in content_config['fields']:
            field_config = FieldConfig()
            content_analyzer_config = ContentAnalyzerConfig(
                runnable_istances[content_config['source_type']](content_config['raw_source_path']),
                content_config['id_field_name'])
            for pipeline_dict in field_dict['pipeline_list']:
                preprocessing_list = list()
                for preprocessing in pipeline_dict['preprocesing_list']:
                    class_name = preprocessing.pop('class')
                    preprocessing_list.append(runnable_istances['class_name'](**preprocessing))
                class_name = pipeline_dict['field_content_production'].pop('class')
                field_config.add_pipeline(FieldRepresentationPipeline(runnable_istances[class_name](**pipeline_dict)),
                                          preprocessing_list)

            content_analyzer = ContentAnalyzer(content_analyzer_config)
            represented_contents = content_analyzer.fit()









config_run()
