from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig
from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import DATFile
from orange_cb_recsys.content_analyzer.config import FieldConfig, FieldRepresentationPipeline
from orange_cb_recsys.content_analyzer.field_content_production_techniques.synset_document_frequency import \
    SynsetDocumentFrequency

movies_filename = '/home/Documents/ml-1m/movies.dat'
user_filename = '/home/Documents/ml-1m/users.dat'
ratings_filename = '/home/Documents/ml-1m/ratings.dat'

output_dir = '../../contents/test_1m_'

movies_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=DATFile(movies_filename),
    id_field_name_list=['0'],
    output_directory=output_dir
)

movies_ca_config.append_field_config(
    field_name='2',         #tag
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            content_technique=SynsetDocumentFrequency())]
    )
)

content_analyzer_movies = ContentAnalyzer(
    config=movies_ca_config
)

content_analyzer_movies.fit()

