import flowcean.cli
from flowcean.core import evaluate_offline, learn_offline
from flowcean.polars import DataFrame
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError
from flowcean.polars.transforms.derivative import Derivative

from flowcean.srlearn.symbolic_regression import SymbolicRegression

def main() -> None:

    flowcean.cli.initialize_logging()
    data = DataFrame.from_uri(uri="file:sampled_data.csv")

    inputs = ["Velocity_t", "dx"]
    outputs = ["Velocity_tplus1"]

    learner = SymbolicRegression(
        features = inputs,
        start_width = 100,
        step_width = 40,
        target_var = outputs,
    )

    model = learn_offline(
        data,
        learner,
        inputs,
        outputs,
    )

if __name__ == "__main__":
    main()