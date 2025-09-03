"""

Symbolic Regression for Smart Sensor 1 Mode Dataset
-----------------------------------------------

This script demostrates how to perform symbolic regression using Flowcean framework.
It loads experimental data from CSV file, trains a SR model, evaluates its performance,
and prints the results.

Key Features:
    - Uses Flowcean's Symbolic Regression for model training.
    - Applies derivative transformations on the target variable.
    - Evaluates the model using Mean Absolute Error (MAE), Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

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
    data = DataFrame.from_uri(uri="file:C:/Users/49157/Desktop/PA/2024---Harshitha-Viswanath---Project-Work/examples/symbolic_regression/smart_sensor_1mode/sampled_data_1mode.csv")
    
    # Define input features and output variables
    inputs = ["time","input_u"]
    outputs = ["output_y"]

    #----------------------------------
    # 3. Configure symbolic regression learner
    #----------------------------------
    learner = SymbolicRegression(
        features = inputs,
        start_width = 100,
        step_width = 100,
        target_var = outputs,
        segmentation_args = {"niterations": 130, 
                             "random_state": 42, 
                             "parsimony":  1.585214534834095e-06,#1.0636201211705178e-05,#0.00032, 
                             "binary_operators":['+', '-', '+'], 
                             "unary_operators": ['sqrt', 'sin', 'cos', 'exp', 'log'],#["sqrt"], 
                             "populations": 49,
                             "model_selection": 'accuracy'},#42},,
        grouping_args = {"niterations": 130, 
                         "random_state": 42, 
                         "parsimony":  1.585214534834095e-06,#1.0636201211705178e-05,#1e-6,  
                         "binary_operators": ['+', '-', '*'], 
                         "unary_operators": ['sqrt', 'sin', 'cos', 'exp', 'log'],#["sqrt"],
                         "populations": 49,
                         "model_selection": 'accuracy'}, #30},
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
        [MeanAbsoluteError(),MeanSquaredError(), RootMeanSquaredError()],
    )

    #----------------------------------
    # 6. Print the evaluation report
    #----------------------------------
    print("\n=== Model Evaluation Report ===")
    print(report)

# Entry point
if __name__ == "__main__":
    main()