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
       diff_cols = [pl.col(col).diff().alias(f"diff_{col}")
                    for col in self.target_var]
       data_frame = data_frame.with_columns(diff_cols)
       
       for col in self.target_var:
           diff_cols = f"diff_{col}"
           val = data_frame[diff_cols][1]
           data_frame = data_frame.with_columns([
                pl.when(pl.arange(0, data_frame.height) == 0)
                .then(val)
                .otherwise(pl.col(diff_cols))
                .alias(diff_cols)])
           
       #print(data_frame)
       
       data_frame = data_frame.drop(*self.target_var)
       data_frame = data_frame.rename({f"diff_{col}": col for col in self.target_var})
       print(data_frame)
       return data_frame
       """ print(data_frame) """

    @override
    def inverse(self):
        return
     