from abc import ABC

import pandas as pd


class Metric(ABC):
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        raise NotImplementedError

