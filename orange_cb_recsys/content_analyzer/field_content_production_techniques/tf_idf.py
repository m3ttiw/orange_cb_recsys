from orange_cb_recsys.content_analyzer.content_representation.content_field import FeaturesBagField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    TfIdfTechnique
from orange_cb_recsys.content_analyzer.memory_interfaces.text_interface import IndexInterface
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.utils.check_tokenization import check_tokenized
from orange_cb_recsys.utils.id_merger import id_merger


class LuceneTfIdf(TfIdfTechnique):
    """
    Class that produces a Bag of words with tf-idf metric
    """

    def __init__(self):
        super().__init__()
        self.__index = IndexInterface('./frequency-index')

    def __str__(self):
        return "LuceneTfIdf"

    def __repr__(self):
        return "< LuceneTfIdf: " + "index = " + str(self.__index) + ">"

    def produce_content(self, field_representation_name: str, content_id: str,
                        field_name: str) -> FeaturesBagField:
        return FeaturesBagField(field_representation_name, self.__index.get_tf_idf(field_name, content_id))

    def dataset_refactor(self, information_source: RawInformationSource, id_field_names: str):
        """
        This method restructures the raw data in a way functional to the final representation.
        This is done only for those field representations that require this phase to be done.
        Args:
            information_source (RawInformationSource):
            id_field_names:

        """

        field_name = self.get_field_need_refactor()
        preprocessor_list = self.get_processor_list()
        pipeline_id = self.get_pipeline_need_refactor()

        self.__index = IndexInterface('./' + field_name + pipeline_id)
        self.__index.init_writing()
        for raw_content in information_source:
            self.__index.new_content()
            content_id = id_merger(raw_content, id_field_names)
            self.__index.new_field("content_id", content_id)
            processed_field_data = raw_content[field_name]
            for preprocessor in preprocessor_list:
                processed_field_data = preprocessor.process(processed_field_data)

            processed_field_data = check_tokenized(processed_field_data)
            self.__index.new_field(field_name, processed_field_data)
            self.__index.serialize_content()

        self.__index.stop_writing()

    def delete_refactored(self):
        self.__index.delete_index()
