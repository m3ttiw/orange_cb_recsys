from orange_cb_recsys.content_analyzer import ContentAnalyzerConfig, FieldRepresentationPipeline, FieldConfig, \
    ContentAnalyzer
from orange_cb_recsys.content_analyzer.field_content_production_techniques import LuceneTfIdf, BabelPyEntityLinking, \
    EmbeddingTechnique, Centroid, Wikipedia2VecDownloader, GensimDownloader
from orange_cb_recsys.content_analyzer.field_content_production_techniques.tf_idf import SkLearnTfIdf
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

import lucene

lucene.initVM(vmargs=['-Djava.awt.headless=true'])

api_key = ''

movies_filename = '../../../datasets/movies_info_reduced.json'

movies_output_dir = '../../../contents/examples/ex_1/movies_'

movies_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=JSONFile(movies_filename),
    id_field_name_list=['imdbID'],
    output_directory=movies_output_dir,

)


movies_ca_config.append_field_config(
    field_name='Title',
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            content_technique=LuceneTfIdf())]
    )
)


movies_ca_config.append_field_config(
    field_name='Year',
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            content_technique=LuceneTfIdf())]
    )
)


movies_ca_config.append_field_config(
    field_name='Genre',
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            preprocessor_list=[NLTK(lemmatization=True, stopwords_removal=True)],
            content_technique=LuceneTfIdf())]
    )
)


movies_ca_config.append_field_config(
    field_name='Plot',
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
                            preprocessor_list=[NLTK(lemmatization=True, stopwords_removal=True)],
                            content_technique=LuceneTfIdf()),
                        FieldRepresentationPipeline(
                            preprocessor_list=[NLTK(lemmatization=True, stopwords_removal=True)],
                            content_technique=EmbeddingTechnique(
                                combining_technique=Centroid(),
                                embedding_source=GensimDownloader(name='text8'),
                                granularity='word')
        )]
    )
)


movies_ca_config.append_field_config(
    field_name='Director',
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            content_technique=BabelPyEntityLinking(api_key=api_key)
        )]
    )
)


movies_ca_config.append_field_config(
    field_name='Actors',
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            content_technique=BabelPyEntityLinking(api_key=api_key)
        )]
    )
)


# visualizza synset di un director

ContentAnalyzer(config=movies_ca_config).fit()