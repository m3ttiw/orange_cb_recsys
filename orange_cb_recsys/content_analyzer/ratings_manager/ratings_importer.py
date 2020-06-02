from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import RatingProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource


class RatingsImporter:
    def __init__(self, source: RawInformationSource,
                 processor: RatingProcessor,
                 user_field_name: str,
                 item_field_name: str,
                 preference_field_name: str,
                 timestamp_field_name: str):

        self.__source: RawInformationSource = source
        self.__processor: RatingProcessor = processor
        self.__user_field_name: str = user_field_name
        self.__item_field_name: str = item_field_name
        self.__preference_field_name: str = preference_field_name
        self.__timestamp_field_name: str = timestamp_field_name

    def import_ratings(self):
        pass


