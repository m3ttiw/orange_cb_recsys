from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import RatingProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
import pandas as pd


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
        self.__columns = ["user_id", "item_id", "original_rating", "derived_score", "timestamp"]

    def get_frame_columns(self) -> list:
        return self.__columns

    def import_ratings(self) -> pd.DataFrame:
        """
        Imports the ratings from the source and stores in a dataframe
        Returns:
            ratings_frame: pd.DataFrame
        """
        ratings_frame = pd.Dataframe(columns=self.__columns)
        for raw_rating in self.__source:
            user_id = raw_rating[self.__user_field_name]
            item_id = raw_rating[self.__item_field_name]
            original_rating = raw_rating[self.__preference_field_name]
            derived_score = self.__processor.fit(original_rating)
            timestamp = raw_rating[self.__timestamp_field_name]
            ratings_frame = ratings_frame.append({
                self.__columns[0]: user_id,
                self.__columns[1]: item_id,
                self.__columns[2]: original_rating,
                self.__columns[3]: derived_score,
                self.__columns[4]: timestamp
            }, ignore_index=True)

        return ratings_frame
