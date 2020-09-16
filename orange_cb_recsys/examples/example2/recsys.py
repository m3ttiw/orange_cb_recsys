from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig, RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from orange_cb_recsys.evaluation import RankingAlgEvalModel, NDCG, FNMeasure, KFoldPartitioning
from orange_cb_recsys.recsys import ClassifierRecommender, RecSysConfig, RecSys

ratings_filename = '../../../datasets/ratings_example.json'
ca_dir = ''

# solo esempio, non presente nel dataset
"""
title_review_config = RatingsFieldConfig(
    field_name='review_title',
    processor=TextBlobSentimentAnalysis()
)
"""

stars_review_config = RatingsFieldConfig(
    field_name='stars',
    processor=NumberNormalizer(min_=1, max_=5))

ratings_importer = RatingsImporter(
    source=JSONFile(ratings_filename),          #cambia
    rating_configs=[stars_review_config],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp',
)

ratings_frame = ratings_importer.import_ratings()

tfidf_classifier_config = ClassifierRecommender(
    item_field='Plot',
    field_representation='0',
    classifier='random_forest'
)

classifier_recsys_config = RecSysConfig(
    users_directory=ca_dir,
    items_directory=ca_dir,
    ranking_algorithm=tfidf_classifier_config,
    rating_frame=ratings_frame
)

classifier_recommender = RecSys(
    config=classifier_recsys_config
)

rank = classifier_recommender.fit_ranking(
    user_id='01',
    recs_number=5
)

print(rank) # non salvare

evaluation_centroid = RankingAlgEvalModel(
    config=classifier_recsys_config,
    partitioning=KFoldPartitioning(n_splits=4),
    metric_list=[NDCG(), FNMeasure(n=2)]
)

results = evaluation_centroid.fit()
print(results)
# aggiungi metriche fairness


wordemb_classifier_config = ClassifierRecommender(
    item_field='Plot',
    field_representation='1',
    classifier='random_forest'
)

classifier_recsys_config.ranking_algorithm(wordemb_classifier_config)

classifier_recommender = RecSys(
    config=classifier_recsys_config
)

rank = classifier_recommender.fit_ranking(
    user_id='01',
    recs_number=5
)

print(rank)  # non salvare

evaluation_centroid = RankingAlgEvalModel(
    config=classifier_recsys_config,
    partitioning=KFoldPartitioning(n_splits=4),
    metric_list=[NDCG(), FNMeasure(n=2)]
)

results = evaluation_centroid.fit()
print(results)
# aggiungi metriche fairness
