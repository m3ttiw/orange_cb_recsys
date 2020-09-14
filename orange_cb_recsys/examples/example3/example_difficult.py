from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile, DATFile
from orange_cb_recsys.evaluation import RankingAlgEvalModel, KFoldPartitioning, Correlation, NDCG
from orange_cb_recsys.recsys import CosineSimilarity, ClassifierRecommender
from orange_cb_recsys.recsys.ranking_algorithms.centroid_vector import CentroidVector
from orange_cb_recsys.recsys.recsys import RecSysConfig

movies_filename = '../../../datasets/movies_info_reduced.json'
user_filename = '../../../datasets/users_info_.json'
ratings_filename = '../../../datasets/ratings_example.json'


output_dir = '../../../contents/test_1m_difficult/dir'


title_review_config = RatingsFieldConfig(
    field_name='review_title',
    processor=TextBlobSentimentAnalysis()
)

starts_review_config = RatingsFieldConfig(
    field_name='stars',
    processor=NumberNormalizer(min_=1, max_=5))

ratings_importer = RatingsImporter(
    source=JSONFile(ratings_filename),
    rating_configs=[title_review_config, starts_review_config],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp',
)

ratings_frame = ratings_importer.import_ratings()


classifier_config = ClassifierRecommender(
    item_field='Plot',
    field_representation='0',
    classifier='random_forest'
)

classifier_recsys_config = RecSysConfig(
    users_directory='../../../contents/test_1m_medium/dir1600079012.1683488',
    items_directory='../../../contents/test_1m_medium/dir1600079332.8693523',
    ranking_algorithm=classifier_config,
    rating_frame=ratings_frame
)

centroid_config = CentroidVector(
    item_field='Director',
    field_representation='0',
    similarity=CosineSimilarity()
)

centroid_recsys_config = RecSysConfig(
    users_directory='../../../contents/test_1m_easy/dir1600074336.5165632',
    items_directory='../../../contents/test_1m_easy/dir1600074336.5165632',
    ranking_algorithm=centroid_config,
    rating_frame=ratings_frame
)

evaluation_classifier = RankingAlgEvalModel(
    config=classifier_recsys_config,
    partitioning=KFoldPartitioning(),
    metric_list=[NDCG(), Correlation(method='spearman')]
)

evaluation_centroid = RankingAlgEvalModel(
    config=centroid_recsys_config,
    partitioning=KFoldPartitioning(),
    metric_list=[NDCG(), Correlation(method='spearman')]
)

eval_frame_classifier = evaluation_classifier.fit()
eval_frame_centroid = evaluation_centroid.fit()
