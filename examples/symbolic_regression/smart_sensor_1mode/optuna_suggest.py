"""
optuna_suggest.py

This script performs hyperparameter optimization for SR 
using Optuna and PySRRegressor to model the smart sensor CPS 
under only one mode of operation: compression mode.
 
Workflow:
    1. Load sampled data from "sampled_data.csv".
    2. Split dataset into training and validation sets.
    3. Use Optuna to optimize hyperparameters.
    4. Train PySR model.
    5. Minimize the mean squared error (MSE) for the model.
    6. Store the optimization study in a local SQLite database.

Inputs:
    - sampled_data.csv -> Preprocessed MATLAB output dataset

Outputs:
    - dcdc.db -> SQLite database storing the Optuna study results
    - Prints the best hyperparameters and validation error.

Dependencies:
    - pandas
    - numpy
    - scikit-learn
    - optuna
    - pysr

Usage:
    python optuna_suggest.py

"""

import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pysr import PySRRegressor
import traceback

# Load data
df = pd.read_csv("sampled_data_1mode.csv")

# Define X and Y
X = df.drop(["time", "output_y"], axis=1).values
Y = df["output_y"].values

def objective(trial):

    """

    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): A single Optuna trial object that
                                    suggests hyperparameters

    Returns:
        float: The combined MSE of both batches

    Workflow:
        1. Sample hyperparameters with range for optimization.
        2. Split dataset into train/validation set(80/20 split).
        3. Train the PySR model.
        4. Calculate MSE for the model and return the value.

    Error Handling:
        If training fails for any reason, MSE with huge penalty of 1e6 is returned. 
    
    """
    
    params = {
        "niterations": trial.suggest_int("niterations", low=50, high=150, step=10),
        "populations": trial.suggest_int("populations", low=30, high=160, step = 1),
        "parsimony": trial.suggest_float("parsimony", low=1e-6, high=0.001, log=True),
        "model_selection": trial.suggest_categorical("model_selection", ["best", "accuracy"]),
        "binary_operators": trial.suggest_categorical("binary_operators", (
                                                      ["+","-"],
                                                      ["+","-","*"],
        )),
        "unary_operators": trial.suggest_categorical("unary_operators",(
                                                     ["sqrt"],
                                                     ["sin","cos"],
        )), 
    }

    # Split data
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize PySR model
    model = PySRRegressor(
        niterations = params["niterations"],
        populations = params["populations"],
        parsimony = params["parsimony"],
        model_selection = params["model_selection"],
        binary_operators = params["binary_operators"],
        unary_operators = params["unary_operators"],
        verbosity = 1,
        random_state = 42,
        deterministic = True,
        parallelism = "serial",
        ncycles_per_iteration = 50,
    )

    try:
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_val)
        val_mse  = mean_squared_error(Y_val, y_pred)
    except Exception as e:
        print(f"\nTrial {trial.number} failed!")
        traceback.print_exc()
    
        val_mse = 1e6 #default val in case of an error
    
    return val_mse
    """ print("Suggested parameters: ", params)
    return 0 """

# Run Optuna
sampler = optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True, group=True)
study = optuna.create_study(storage="sqlite:///ss1mode.db",
                            direction="minimize",
                            sampler=sampler)
study.optimize(objective, n_trials=60)

print("Best parameters: ", study.best_params)
print(f"Best validation MSE: {study.best_value: .5f}")