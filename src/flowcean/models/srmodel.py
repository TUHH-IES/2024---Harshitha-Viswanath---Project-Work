from pathlib import Path
from typing import Any
import polars as pl

# Base Model class (if defined in your project)
from flowcean.core.model import Model

class SymbolicRegressionModel(Model):
    def __init__(self, results: list[pl.DataFrame]) -> None:
        self.results = results

    def predict(self, input_features: pl.DataFrame) -> pl.DataFrame:
        input_features = input_features.with_row_count(name="count")
        grouping_windows = self.results[0]
        grouping_results = self.results[1]

        # Start with an 'id' column set to None
        output = input_features.with_columns(pl.lit(None).alias("group_id"))

        # Iterate over grouping_windows to assign group IDs
        for row in grouping_windows.iter_rows():
            group_id, window_start, window_end = row

            output = output.with_columns(
                pl.when((pl.col("count") >= window_start) & (pl.col("count") <= window_end))
                .then(group_id)
                .otherwise(pl.col("group_id"))
                .alias("group_id")
            )

        output = output.with_columns(
            (pl.col("group_id") != pl.col("group_id").shift(1)).cum_sum().alias("block_id")
        )

        result = (
            output.group_by("group_id", "block_id")
            .agg([
                pl.col("count").min().alias("start_window"),
                pl.col("count").max().alias("end_window")
            ])
        )

        result = result.filter(pl.col("block_id").is_not_null())
        result = result.drop("block_id")

        result = result.with_columns(pl.col("group_id").cast(pl.Int64))
        result = result.join(grouping_results, on = "group_id", how = "inner") #inner join: returns all the rows matched on both dataframe
        #left join: returns all the rows from left df and only matched rows from right df. full join: returns all the rows from both dfs with null values for missing common col rows

        pl.Config.set_tbl_rows(3000)  # Set maximum rows to print
        pl.Config.set_tbl_cols(100) 
        print(result)
        return output

    def load(self, path: Path) -> None:   #What to implement here since there is no .pkl to load and dump
        self.results[0] = pl.read_csv(path)

    def save(self, path: Path) -> None:
        self.results[1] = pl.read_csv(path)


def main():
    # Paths to input files
    grouping_windows_path = Path(
        "C:/Users/49157/Desktop/PA/SR_Original_code/SymbolicRegression4HA/results/converter/grouping_windows.csv"
    )
    grouping_results_path = Path(
        "C:/Users/49157/Desktop/PA/SR_Original_code/SymbolicRegression4HA/results/converter/grouping_results.csv"
    )

    # Load the data into Polars DataFrames
    grouping_windows = pl.read_csv(grouping_windows_path)
    grouping_results = pl.read_csv(grouping_results_path)

    # Initialize the SymbolicRegressionModel
    model = SymbolicRegressionModel([grouping_windows, grouping_results])

    # Predict on new input data
    input_path = Path(
        "C:/Users/49157/Desktop/PA/SR_Original_code/SymbolicRegression4HA/data/converter/short_wto_zeros_data_converter_omega400e3_beta40e3_Q10_theta60.csv"
    )
    input_data = pl.read_csv(input_path)

    # Call the predict method
    model.predict(input_data)

    """ model.save(grouping_results_path)  # Save the model
    model.load(grouping_windows_path)  # Load the model
 """

if __name__ == "__main__":
    main()
