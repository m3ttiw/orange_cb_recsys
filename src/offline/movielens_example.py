from src.offline.content_analyzer.content_analyzer_main import FieldConfig, ContentAnalyzerConfig, \
    FieldRepresentationPipeline, ContentAnalyzer
from src.offline.content_analyzer.embedding_source import BinaryFile
from src.offline.content_analyzer.entity_linking import BabelPyEntityLinking
from src.offline.content_analyzer.field_content_production_technique import TfIdfTechnique, EmbeddingTechnique, \
    Granularity
from src.offline.content_analyzer.nlp import OpenNLP
from src.offline.memory_interfaces.text_interface import IndexInterface
from src.offline.raw_data_extractor.raw_data_manager import RawDataConfig, RawDataManager
from src.offline.raw_data_extractor.raw_information_source import JSONFile

print("FASE 1")
print("##################################################")

config_dict = {"Plot": IndexInterface('./test-index-plot')}
raw_data_config = RawDataConfig(JSONFile("movies_info.json"), "imdbID", config_dict)
raw_data_manager = RawDataManager(raw_data_config).fit()

print("FASE 2")
print("##################################################")

field_config = FieldConfig()
content_analyzer_config = ContentAnalyzerConfig(JSONFile("movies_info.json"), "imdbID")

title_content_pipeline_EL = FieldRepresentationPipeline(BabelPyEntityLinking(), None)

field_config.add_pipeline(title_content_pipeline_EL)

plot_content_pipeline_tf_idf = FieldRepresentationPipeline(TfIdfTechnique(raw_data_config.get_interface("plot"),),
                                                           [OpenNLP(stopwords_removal=True, lemmatization=True)])

field_config.add_pipeline(plot_content_pipeline_tf_idf)

plot_content_pipeline_embedding = FieldRepresentationPipeline(EmbeddingTechnique(None, BinaryFile("example_name"),
                                                              Granularity.WORD),
                                                              [OpenNLP(url_tagging=True,
                                                                       strip_multiple_whitespaces=False)])

content_analyzer_config.append_field_config("plot", field_config)
content_analyzer = ContentAnalyzer(content_analyzer_config)
represented_contents = content_analyzer.fit()
