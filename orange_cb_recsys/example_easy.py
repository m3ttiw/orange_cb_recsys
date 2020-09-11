from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig
from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import DATFile, JSONFile
from orange_cb_recsys.content_analyzer.config import FieldConfig, FieldRepresentationPipeline
from orange_cb_recsys.content_analyzer.field_content_production_techniques.synset_document_frequency import \
    SynsetDocumentFrequency
from orange_cb_recsys.recsys import CosineSimilarity
from orange_cb_recsys.recsys.recsys import RecSys, RecSysConfig
from orange_cb_recsys.recsys.ranking_algorithms.centroid_vector import CentroidVector

import pandas as pd

movies_filename = '../../datasets/movies_info_reduced.json'
ratings_filename = '../../datasets/test_import_ratings.json'

output_dir = '../../contents/test_1m_'


movies_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=JSONFile(movies_filename),
    id_field_name_list=['imdbID'],
    output_directory=output_dir
)


movies_ca_config.append_field_config(
    field_name='Director',         #tag
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            content_technique=SynsetDocumentFrequency())]
    )
)


content_analyzer_movies = ContentAnalyzer(
    config=movies_ca_config
)

content_analyzer_movies.fit()


ratings_importer = RatingsImporter(
    source=JSONFile(ratings_filename),
    rating_configs=[RatingsFieldConfig(
        field_name='stars',
        processor=NumberNormalizer(min_=1, max_=5))],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp',
)

ratings_frame = ratings_importer.import_ratings()


centroid_config = CentroidVector(
    item_field='Director',
    field_representation='0',
    similarity=CosineSimilarity()
)


centroid_recsys_config = RecSysConfig(
    users_directory=output_dir,
    items_directory=output_dir,
    ranking_algorithm=centroid_config,
    rating_frame=ratings_frame
)


centroid_recommender = RecSys(
    config=centroid_recsys_config
)

centroid_recommender.fit_ranking(
    user_id='1',
    recs_number=10
)
