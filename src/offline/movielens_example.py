from offline.content_analyzer.content_analyzer_main import ContentAnalyzerConfig, ContentAnalyzer, FieldContentPipeline
from offline.content_analyzer.embedding_source import BinaryFile
from offline.content_analyzer.entity_linking import BabelPyEntityLinking
from offline.content_analyzer.field_content_production_technique import TfIdfTechnique, EmbeddingTechnique, Granularity
from offline.memory_interfaces.text_interface import IndexInterface
from offline.content_analyzer.nlp import OpenNLP
from offline.raw_data_extractor.raw_data_manager import RawFieldPipeline, RawDataConfig, RawDataManager
from offline.raw_data_extractor.raw_information_source import JSONFile
from offline.utils.id_utils import extract_ids

items_id_list = extract_ids("D:\\Users\\robb\\Desktop\\ml-1m\\movies.dat", 0, "::", 10, 20)

print("FASE 1")
print("##################################################")

title_raw_data_pipeline = RawFieldPipeline(JSONFile("example_path"), IndexInterface("example_output_directory"))
config_dict = {"title": title_raw_data_pipeline}
plot_raw_data_pipeline = RawFieldPipeline(JSONFile("example_path"), IndexInterface("example_output_directory"))
config_dict["plot"] = plot_raw_data_pipeline
raw_data_config = RawDataConfig(config_dict)
raw_data_manager = RawDataManager(items_id_list, raw_data_config).fit()

print("FASE 2")
print("##################################################")

content_analyzer_config = ContentAnalyzerConfig()

title_content_pipeline_EL = FieldContentPipeline(raw_data_config.get_pipeline("title").get_memory_interface(),
                                                 BabelPyEntityLinking(),
                                                 None)

content_analyzer_config.add_pipeline("title", title_content_pipeline_EL)

plot_content_pipeline_tf_idf = FieldContentPipeline(raw_data_config.get_pipeline("plot").get_memory_interface(),
                                                    TfIdfTechnique(
                                                        raw_data_config.get_pipeline("plot").get_memory_interface(),
                                                        items_id_list),
                                                    [OpenNLP(stopwords_removal=True, lemmatization=True)])

content_analyzer_config.add_pipeline("plot", plot_content_pipeline_tf_idf)

plot_content_pipeline_embedding = FieldContentPipeline(raw_data_config.get_pipeline("plot").get_memory_interface(),
                                                       EmbeddingTechnique(None, BinaryFile("example_name"),
                                                                          Granularity.WORD),
                                                       [OpenNLP(url_tagging=True, strip_multiple_whitespaces=False)])

content_analyzer_config.add_pipeline("plot", plot_content_pipeline_embedding)
content_analyzer = ContentAnalyzer(items_id_list, content_analyzer_config)
represented_items = content_analyzer.fit()
