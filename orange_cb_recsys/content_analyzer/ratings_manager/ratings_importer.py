from typing import List

from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import RatingProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
import pandas as pd


class RatingsFieldConfig:
    def __init__(self, preference_field_name: str,
                 processor: RatingProcessor):
        self.__preference_field_name = preference_field_name
        self.__processor = processor

    def get_field_name(self):
        return self.__preference_field_name

    def get_processor(self):
        return self.__processor


class RatingsImporter:
    def __init__(self, source: RawInformationSource,
                 rating_configs: List[RatingsFieldConfig],
                 from_field_name: str,
                 to_field_name: str,
                 timestamp_field_name: str):

        self.__source: RawInformationSource = source
        self.__rating_configs: List[RatingsFieldConfig] = rating_configs
        self.__from_field_name: str = from_field_name
        self.__to_field_name: str = to_field_name
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
            user_id = raw_rating[self.__from_field_name]
            item_id = raw_rating[self.__to_field_name]
            timestamp = raw_rating[self.__timestamp_field_name]
            for preference in self.__rating_configs:
                original_rating = raw_rating[preference.get_field_name()]
                derived_score = preference.get_processor().fit(original_rating)
                ratings_frame = ratings_frame.append({
                    self.__columns[0]: user_id,
                    self.__columns[1]: item_id,
                    self.__columns[2]: original_rating,
                    self.__columns[3]: derived_score,
                    self.__columns[4]: timestamp
                }, ignore_index=True)

        return ratings_frame  # si potrebbe memorizzare in un output_dierctory a scelta dell'utente
