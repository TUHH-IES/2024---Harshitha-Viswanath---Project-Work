"""

Symbolic Regression for DC-DC Converter Dataset
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
from flowcean.models.srmodel import SymbolicRegressionModel

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
    data = DataFrame.from_uri(uri="file:C:/Users/49157/Desktop/PA/2024---Harshitha-Viswanath---Project-Work/examples/symbolic_regression/dcdc_converter/sampled_data.csv")

    # Define input features and output variables
    inputs = ["CapacitorVoltage_V", "Vin_V", "L_H"]
    outputs = ["InductorCurrent_A"]

    #----------------------------------
    # 3. Apply derivative transformation to outputs
    #----------------------------------
    derivative = Derivative(outputs)

    #----------------------------------
    # 4. Configure symbolic regression learner
    #----------------------------------
    learner = SymbolicRegression(
        features = inputs,
        start_width = 100,
        step_width = 40,
        target_var = outputs,
        segmentation_args = {
                            "niterations": 150,#60,#50,
                            "random_state": 42, 
                            "parsimony": 0.00016956553008917993,#7.636825749110554e-05,#0.00032, 
                            "binary_operators":["+", "-", "*"], 
                            "unary_operators": ['sqrt', 'sin', 'cos', 'exp', 'log'],#["sqrt"], 
                            "populations": 48},#57},#42},}
        grouping_args = {
                        "random_state": 42, 
                        "parsimony": 0.00016956553008917993,#7.636825749110554e-05,#1e-6,  
                        "binary_operators": ["+", "-", "*"], 
                        "unary_operators": ['sqrt', 'sin', 'cos', 'exp', 'log'],#["sqrt"],
                        "populations": 57},#30},
    )
    
    #----------------------------------
    # 5. Train the model
    #----------------------------------
    model = learn_offline(
        data,
        learner,
        inputs,
        outputs,
        output_transform = derivative,
    ) 

    #----------------------------------
    # 6. Evaluate the model
    #----------------------------------

    report = evaluate_offline(
    model,
    data,
    inputs,
    outputs,
    [MeanSquaredError(), RootMeanSquaredError()],
    ) 
    
    #----------------------------------
    # 7. Print the evaluation report
    #----------------------------------
    print("\n=== Model Evaluation Report ===")
    print(report)

# Entry point
if __name__ == "__main__":
    main() 