from typing import List
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import RatingProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.content_analyzer.ratings_manager.score_combiner import ScoreCombiner
import pandas as pd
import time


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
                 timestamp_field_name: str,
                 output_directory: str = None,
                 score_combiner: str = "avg"):

        self.__source: RawInformationSource = source
        self.__output_directory: str = output_directory
        self.__rating_configs: List[RatingsFieldConfig] = rating_configs
        self.__from_field_name: str = from_field_name
        self.__to_field_name: str = to_field_name
        self.__timestamp_field_name: str = timestamp_field_name
        self.__score_combiner = ScoreCombiner(score_combiner)

        self.__columns: list = ["from_id", "to_id", "score", "timestamp"]
        for i, field in enumerate(self.__rating_configs):
            self.__columns.append(field.get_field_name())

    def get_frame_columns(self) -> list:
        return self.__columns

    def get_from_field_name(self) -> str:
        return self.__from_field_name

    def get_to_field_name(self) -> str:
        return self.__to_field_name

    def get_timestamp_field_name(self) -> str:
        return self.__timestamp_field_name

    def import_ratings(self) -> pd.DataFrame:
        """
        Imports the ratings from the source and stores in a dataframe
        Returns:
            ratings_frame: pd.DataFrame
        """
        ratings_frame = pd.DataFrame(columns=list(self.__columns))
        for raw_rating in self.__source:
            score_list = []
            row_dict = {
                "from_id": raw_rating[self.__from_field_name],
                "to_id": raw_rating[self.__to_field_name],
                "timestamp": raw_rating[self.__timestamp_field_name],
            }
            for i, preference in enumerate(self.__rating_configs):
                row_dict[preference.get_field_name()] = raw_rating[preference.get_field_name()]
                score_list.append(preference.get_processor().fit(row_dict[preference.get_field_name()]))

            row_dict["score"] = self.__score_combiner.combine(score_list)
            ratings_frame = ratings_frame.append(row_dict, ignore_index=True)

        if self.__output_directory is not None:
            try:
                ratings_frame.to_csv("../../../datasets/{}/ratings_{}.csv".format(self.__output_directory,
                                                                                  time.time()), index=False, header=False)
            except FileNotFoundError:
                ratings_frame.to_csv("datasets/{}/ratings_{}.csv".format(self.__output_directory,
                                                                         time.time()), index=False, header=False)

        return ratings_frame
