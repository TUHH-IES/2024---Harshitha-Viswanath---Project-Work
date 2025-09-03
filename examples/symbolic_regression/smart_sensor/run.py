"""

Symbolic Regression for Smart Sensor Dataset
-----------------------------------------------

This script demostrates how to perform symbolic regression using Flowcean framework.
It loads experimental data from CSV file, trains a SR model, evaluates its performance,
and prints the results.

Key Features:
    - Uses Flowcean's Symbolic Regression for model training.
    - Applies derivative transformations on the target variable.
    - Evaluates the model using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

"""

import flowcean.cli
from flowcean.core import evaluate_offline, learn_offline
from flowcean.polars import DataFrame
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from flowcean.polars.transforms.derivative import Derivative

from flowcean.srlearn.symbolic_regression import SymbolicRegression

def main() -> None:

    """

    Main execution function.
    Loads the dataset, defines a SR model, and trains it, 
    evaluates its performance, and prints the evaluation report
    
    """

    #----------------------------------
    # 1. Initialize logging
    #----------------------------------
    flowcean.cli.initialize_logging()

    #----------------------------------
    # 2. Load dataset
    #----------------------------------
    data = DataFrame.from_uri(uri="file:sampled_data.csv")
    
    # Define input features and output variables
    inputs = ["input_u"]
    outputs = ["output_y"]

    #----------------------------------
    # 3. Configure symbolic regression learner
    #----------------------------------
    learner = SymbolicRegression(
        features = inputs,
        start_width = 100,
        step_width = 20,
        target_var = outputs,
        segmentation_args = {"niterations": 120,#60,#25,
                            "random_state": 42, 
                            "parsimony": 5.15846995941542e-05,#0.0006266631861846377,#0.1, 
                            "binary_operators":["+", "-", "*"], 
                            "unary_operators": ['sqrt', 'sin', 'cos'],#["sqrt"], 
                            "populations": 48,
                            "model_selection": 'best'},#53},#42},
        grouping_args = {"random_state": 42, 
                        "parsimony": 5.15846995941542e-05,#0.0006266631861846377, #0.1, 
                        "binary_operators": ["+", "-", "*"], 
                        "unary_operators": ['sqrt', 'sin', 'cos'],#["sqrt"],
                        "populations": 48,
                        "model_selection": 'best'},#53},#30},
    )

    #----------------------------------
    # 4. Train the model
    #----------------------------------
    model = learn_offline(
        data,
        learner,
        inputs,
        outputs,
    )

    #----------------------------------
    # 5. Evaluate the model
    #----------------------------------
    report = evaluate_offline(
    model,
    data,
    inputs,
    outputs,
    [MeanSquaredError(), RootMeanSquaredError()],
    ) 
    
    #----------------------------------
    # 6. Print the evaluation report
    #----------------------------------
    print("\n=== Model Evaluation Report ===")
    print(report)

# Entry point
if __name__ == "__main__":
    main()