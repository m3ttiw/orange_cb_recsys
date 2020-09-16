from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from orange_cb_recsys.evaluation import RankingAlgEvalModel, KFoldPartitioning, NDCG, ReportEvalModel, FNMeasure
from orange_cb_recsys.recsys import CentroidVector, CosineSimilarity, RecSysConfig, RecSys

ratings_filename = '../../../datasets/ratings_example.json'

ca_dir = '../../../contents/examples/ex_1/movies_1600270629.4642215'

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

print(ratings_frame)


centroid_config = CentroidVector(
    item_field='Plot',
    field_representation='0',
    similarity=CosineSimilarity()
)


centroid_recsys_config = RecSysConfig(
    users_directory=ca_dir,
    items_directory=ca_dir,
    ranking_algorithm=centroid_config,
    rating_frame=ratings_frame
)


centroid_recommender = RecSys(
    config=centroid_recsys_config
)

rank = centroid_recommender.fit_ranking(
    user_id='01',
    recs_number=5
)

print(rank)

# fa vedere come salvare il rank su un file csv

evaluation_centroid = RankingAlgEvalModel(
    config=centroid_recsys_config,
    partitioning=KFoldPartitioning(n_splits=4),
    metric_list=[NDCG(), FNMeasure(n=2)]
)

results = evaluation_centroid.fit()

print(results)

