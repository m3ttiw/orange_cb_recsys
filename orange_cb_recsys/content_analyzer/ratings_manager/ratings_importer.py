from typing import List
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import RatingProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.content_analyzer.ratings_manager.score_combiner import ScoreCombiner
import pandas as pd
import time
import logging
from orange_cb_recsys.utils.const import home_path, logger, DEVELOPING


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
        self.__file_name: str = output_directory
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
        logging.basicConfig(level=logging.INFO)
        ratings_frame = pd.DataFrame(columns=list(self.__columns))

        dicts = \
            [
                {
                    **{
                        "from_id": raw_rating[self.__from_field_name],
                        "to_id": raw_rating[self.__to_field_name],
                        "timestamp": raw_rating[self.__timestamp_field_name],
                        "score": self.__score_combiner.combine(
                            [preference.get_processor().fit(raw_rating[preference.get_field_name()])
                             for preference in self.__rating_configs])
                    },
                    **{
                        preference.get_field_name(): raw_rating[preference.get_field_name()] for preference in
                        self.__rating_configs
                    }
                }
                for raw_rating in show_progress(self.__source)
            ]

        ratings_frame = ratings_frame.append(dicts, ignore_index=True)

        if self.__file_name is not None:
            if not DEVELOPING:
                ratings_frame.to_csv("{}/ratings/{}_{}.csv".format(home_path, self.__file_name, int(time.time())),
                                     index=False, header=True)
            else:
                ratings_frame.to_csv("{}_{}.csv".format(self.__file_name, int(time.time())), index=False, header=True)

        return ratings_frame


def show_progress(coll, milestones=100):
    processed = 0
    for x in coll:
        yield x
        processed += 1
        if processed % milestones == 0:
            logger.info('Processed %s elements' % processed)
