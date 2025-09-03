"""
optuna_suggest.py

This script performs hyperparameter optimization for SR 
using Optuna and PySRRegressor to model the inductor current 
in a DC-DC buck converter CPS.
 
Workflow:
    1. Load sampled data from "sampled_data.csv".
    2. Split dataset into training and validation sets.
    3. Use Optuna to optimize hyperparameters for two seperate batches:
        - close_train and close_val -> SwitchState = 1, when switch is closed
        - open_train and open_val -> SwictState = 0, when switch is open
    4. Train PySR models for both batches.
    5. Minimize the combined mean squared error (MSE) for both batches.
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


# Load data
df = pd.read_csv("sampled_data.csv")

# Define X and Y
X = df.drop(["InductorCurrent_A", "Time_s", "SwitchState", "C_F", "R_Ohm"], axis=1).values
Y = df["InductorCurrent_A"].values
class_labels = df["SwitchState"].values  # Keep class info separately

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
        3. Train two seperate PySR models:
            - model_close -> closed switchstate(SwitchState=1)
            - model_open -> open switchstate(SwitchState=0)
        4. Calculate MSE for both the models and return the weighted combined error.

    Error Handling:
        If training fails for any reason, MSE with huge penalty of 1e6 is returned. 
    
    """
    params = {
        "niterations": trial.suggest_int("niterations", low=50, high=150, step=10),
        "populations": trial.suggest_int("populations", low=42, high=60, step=1),
        "parsimony": trial.suggest_float("parsimony", low=1e-6, high=0.001, log=True),
        "model_selection": trial.suggest_categorical("model_selection", ["best", "accuracy"]),
        "binary_operators": trial.suggest_categorical("binary_operators", [
            ["+", "-"],
            ["+", "-", "*"],
            ["+", "-", "*", "/"],
        ]),
        "unary_operators": trial.suggest_categorical("unary_operators", [
            ["sqrt"],
            ["sqrt", "sin", "cos"],
            ["sqrt", "sin", "cos", "exp"],
            ["sqrt", "sin", "cos", "exp", "log"],
        ]),
    }

    # Split data
    X_train, X_val, Y_train, Y_val, class_train, class_val = train_test_split(
        X, Y, class_labels, test_size=0.2, random_state=42
    )

    # Masks for class
    close_train = class_train == 1 # close_train is 1 where class_train is 1: samples where switch is closed
    open_train = class_train == 0 #samples where switch is open
    close_val = class_val == 1
    open_val = class_val == 0

    # Initialize models for both batches
    model_close = PySRRegressor(
        niterations=params["niterations"],
        populations=params["populations"],
        parsimony=params["parsimony"],
        model_selection=params["model_selection"],
        binary_operators=params["binary_operators"],
        unary_operators=params["unary_operators"],
        verbosity=0,
        random_state=42,
        deterministic=True,
        parallelism="serial", #procs=0 no more valid. If used, Optuna fails to optimize
        ncycles_per_iteration=50,
    )

    model_open = PySRRegressor(
        niterations=params["niterations"],
        populations=params["populations"],
        parsimony=params["parsimony"],
        model_selection=params["model_selection"],
        binary_operators=params["binary_operators"],
        unary_operators=params["unary_operators"],
        verbosity=0,
        random_state=42,
        deterministic=True,
        parallelism="serial", #procs=0 no more valid. If used, Optuna fails to optimize
        ncycles_per_iteration=50,
    )

    try:
        model_close.fit(X_train[close_train], Y_train[close_train])
        model_open.fit(X_train[open_train], Y_train[open_train])

        y_pred_close = model_close.predict(X_val[close_val])
        y_pred_open = model_open.predict(X_val[open_val])

        val_mse_close = mean_squared_error(Y_val[close_val], y_pred_close)
        val_mse_open = mean_squared_error(Y_val[open_val], y_pred_open)

        combined_error = (
            val_mse_close * len(y_pred_close) + val_mse_open * len(y_pred_open)
        ) / len(Y_val)
    except Exception as e:
        print("Error during model training:", str(e))
        combined_error = 1e6

    return combined_error

# Run Optuna
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(#sampler=optuna.samplers.CmaEsSampler(seed=42),
                            storage="sqlite:///dcdc.db",
                            direction="minimize",
                            pruner=pruner)
study.optimize(objective, n_trials=20)

print("Best parameters: ", study.best_params)
print(f"Best validation MSE: {study.best_value:.5f}")




