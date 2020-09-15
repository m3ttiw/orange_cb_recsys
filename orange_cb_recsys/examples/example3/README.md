## Example 3

In this example we will focus our attention on the recommendation phase and the creation of a valid evaluation model. For example, I need to compare the performance of a random forest classifier recommender and a centroid vector recommender. We will use the contents already created in the two previus examples.

So first we need to import some framework parts:

```
from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from orange_cb_recsys.evaluation import RankingAlgEvalModel, KFoldPartitioning, Correlation, NDCG
from orange_cb_recsys.recsys import CosineSimilarity, ClassifierRecommender
from orange_cb_recsys.recsys.ranking_algorithms.centroid_vector import CentroidVector
from orange_cb_recsys.recsys.recsys import RecSysConfig
```

Next we define the dir constants, for this example we will use some pre-created contents from the two previus example as we can see in the code:
```
movies_filename = 'items_dir'           # change this with your directory
user_filename = 'users_dir'             # change this with your directory
ratings_filename = '../../../datasets/ratings_example.json'

```

Let's move on to the ratings part: We instantiate two RatingsFieldConfig, one that carries out the Sentiment Analysis on the title field of the review and the other on the rating (stars) as in the previous example:

```
title_review_config = RatingsFieldConfig(
    field_name='review_title',
    processor=TextBlobSentimentAnalysis()
)

starts_review_config = RatingsFieldConfig(
    field_name='stars',
    processor=NumberNormalizer(min_=1, max_=5))
```

We then instantiate the ratings Importer which returns the ratings frame using the "import_ratings ()" method:

```
ratings_importer = RatingsImporter(
    source=JSONFile(ratings_filename),
    rating_configs=[title_review_config, starts_review_config],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp',
)

ratings_frame = ratings_importer.import_ratings()
```

Now let's move on to the recommending part: we instantiate a classifier with the random forest technique, as in the previous example:

```
classifier_config = ClassifierRecommender(
    item_field='Plot',
    field_representation='0',
    classifier='random_forest'
)
```

The newly created config is then passed to the RecSysConfig:

```
classifier_recsys_config = RecSysConfig(
    users_directory=users_filename,
    items_directory=movies_filename,
    ranking_algorithm=classifier_config,
    rating_frame=ratings_frame
)
```

We instantiate a "CentroidVector" object, to which the following parameters will be passed, as in example 1:

```
centroid_config = CentroidVector(
    item_field='Director',
    field_representation='0',
    similarity=CosineSimilarity()
)
```

The newly created config is passed to a new RecSysConfig object:

```
centroid_recsys_config = RecSysConfig(
    users_directory=users_filename,
    items_directory=movies_filename,
    ranking_algorithm=centroid_config,
    rating_frame=ratings_frame
)
```

Let's move on to the Evaluation part, creating a "RankingAlvEvalModel" object, to which the previously instantiated classifier config will be passed, the "KFoldPartitioning" object to the "partitioning" parameter (which allows to use the KFold cross validation technique), and a list of metrics to use (in this case the Spearman Correlation Coefficient and the Discounted Cumulative Gain):

```
evaluation_classifier = RankingAlgEvalModel(
    config=classifier_recsys_config,
    partitioning=KFoldPartitioning(),
    metric_list=[NDCG(), Correlation(method='spearman')]
)
```

Same thing as the previous image, in this case passing the "CentroidVector" config created previously:

```
evaluation_centroid = RankingAlgEvalModel(
    config=centroid_recsys_config,
    partitioning=KFoldPartitioning(),
    metric_list=[NDCG(), Correlation(method='spearman')]
)
```

Finally we train the two "RankingAlgEvalModel" objects just created using the "fit ()" method:

```
eval_frame_classifier = evaluation_classifier.fit()
eval_frame_centroid = evaluation_centroid.fit()
```
