from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from orange_cb_recsys.recsys import NXPageRank
from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
from orange_cb_recsys.utils.feature_selection import NXFSPageRank

ratings_filename = '../../../datasets/ratings_example.json'

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
)

# aggiungi metrica semplice

print(rank)


rank = NXPageRank(graph=full_graph).predict(
    user_id='01',
    ratings=ratings_import,
    recs_number=10,
    feature_selection_algorithm=NXFSPageRank()
)

# aggiungi metrica semplice

print(rank)

