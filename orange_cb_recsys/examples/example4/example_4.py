from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig
from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import DATFile, JSONFile
from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique, \
    PropertiesFromDataset

from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
from orange_cb_recsys.recsys.ranking_algorithms import NXPageRank
from orange_cb_recsys.utils.feature_selection import NXFSPageRank

from orange_cb_recsys.evaluation.graph_metrics import nx_degree_centrality, nx_dispersion


movies_filename = '../../../datasets/movies_info_reduced.json'
user_filename = '../../../datasets/users_info_.json'
ratings_filename = '../../../datasets/ratings_example.json'


output_dir_movies = '../../../contents/test_1m_ex_4/movies_'
output_dir_users = '../../../contents/test_1m_ex_4/users_'
"""
movies_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=JSONFile(movies_filename),
    id_field_name_list=['imdbID'],
    output_directory=output_dir_movies
)


movies_ca_config.append_exogenous_properties_retrieval(
    DBPediaMappingTechnique(
        entity_type='Film',
        lang='EN',
        label_field='Title'
    )
)


content_analyzer = ContentAnalyzer(movies_ca_config).fit()


users_ca_config = ContentAnalyzerConfig(
    content_type='User',
    source=JSONFile(user_filename),
    id_field_name_list=['user_id'],
    output_directory=output_dir_users
)


users_ca_config.append_exogenous_properties_retrieval(
    PropertiesFromDataset()
)

content_analyzer = ContentAnalyzer(users_ca_config).fit()

"""
ratings_import = RatingsImporter(
    source=JSONFile(ratings_filename),
    rating_configs=[RatingsFieldConfig(field_name='stars', processor=NumberNormalizer(min_=1, max_=5))],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp'
).import_ratings()


full_graph = NXFullGraph(
    source_frame=ratings_import,
    contents_dir='../../../contents/test_1m_ex_4/users_',
    user_exogenous_properties=None,
    item_exogenous_properties=['director', 'protagonist', 'producer']
)


rank = NXPageRank(graph=full_graph).predict(
    user_id='01',
    ratings=ratings_import,
    recs_number=10,
    feature_selection_algorithm=NXFSPageRank()
)

print(rank)

#print(nx_dispersion(full_graph))
#print(nx_degree_centrality(full_graph))
