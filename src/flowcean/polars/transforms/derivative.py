import logging
from collections.abc import Iterable
from ruamel.yaml import YAML

import polars as pl
from polars._typing import IntoExpr
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)

class Derivative:

    def __init__(self, target_var):
        self.target_var = target_var

    @override
    def apply(self, output_features) -> pl.LazyFrame:
       data_frame = output_features.collect()
       data_frame = data_frame.with_columns(diff=pl.col(self.target_var).diff())
       data_frame[0, "diff"] = data_frame["diff"][1]
       data_frame = data_frame.drop(self.target_var[0])
       data_frame = data_frame.with_columns(pl.col("diff").alias(self.target_var[0])).drop("diff")
       return data_frame
       """ print(data_frame) """

    @override
    def inverse(self):
        return
     