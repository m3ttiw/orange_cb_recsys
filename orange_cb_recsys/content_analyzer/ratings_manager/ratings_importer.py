from abc import ABC, abstractmethod
from typing import List
import time
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import RatingProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.content_analyzer.ratings_manager.score_combiner import ScoreCombiner
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
                 output_directory: str,
                 rating_configs: List[RatingsFieldConfig],
                 from_field_name: str,
                 to_field_name: str,
                 timestamp_field_name: str,
                 score_combiner: str = "avg"):

        self.__source: RawInformationSource = source
        self.__output_directory: str = output_directory
        self.__rating_configs: List[RatingsFieldConfig] = rating_configs
        self.__from_field_name: str = from_field_name
        self.__to_field_name: str = to_field_name
        self.__timestamp_field_name: str = timestamp_field_name
        self.__score_combiner = ScoreCombiner(eval(score_combiner))

        self.__columns: dict = {
            "from_id": self.__from_field_name,
            "to_id": self.__to_field_name,
            "score": "score",
            "timestamp": self.__timestamp_field_name}
        for i, field in enumerate(self.__rating_configs):
            self.__columns["original_rating_{}".format(i)] = field.get_field_name()

    def get_frame_columns(self) -> dict:
        return self.__columns

    def import_ratings(self) -> pd.DataFrame:
        """
        Imports the ratings from the source and stores in a dataframe
        Returns:
            ratings_frame: pd.DataFrame
        """
        ratings_frame = pd.Dataframe(columns=self.__columns.values())
        for raw_rating in self.__source:
            score_list = []
            row_dict = {
                self.__columns["from_id"]: raw_rating[self.__from_field_name],
                self.__columns["to_id"]: raw_rating[self.__to_field_name],
                self.__columns["timestamp"]: raw_rating[self.__timestamp_field_name],
            }
            for i, preference in enumerate(self.__rating_configs):
                row_dict["original_rating_{}".format(i)] = raw_rating[preference.get_field_name()]
                score_list.append(preference.get_processor().fit(row_dict["original_rating_{}".format(i)]))

            row_dict[self.__columns["score"]] = self.__score_combiner.combine(score_list)
            ratings_frame = ratings_frame.append(row_dict, ignore_index=True)

        ratings_frame.to_csv("{}/ratings_{}".format(self.__output_directory, time.time), index=False, header=False)

        return ratings_frame
