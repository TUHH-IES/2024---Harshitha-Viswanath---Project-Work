"""
optuna_suggest.py

This script performs hyperparameter optimization for SR 
using Optuna and PySRRegressor to model the vehicle cruise 
control CPS.
 
Workflow:
    1. Load sampled data from "sampled_data.csv".
    2. Split dataset into training and validation sets.
    3. Use Optuna to optimize hyperparameters for two seperate batches:
        - class_accelerate_train and class_accelerate_val -> data of vehicle accelerating
        - class_brake_train and class_brake_val -> data of vehicle braking
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
mode = df["Mode"].values
X = df.drop(["Velocity_tplus1", "Time_s", "Mode"], axis=1).values
Y = df["Velocity_tplus1"].values
class_id = -1

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
            - model_accelerate -> vehicle accelerating
            - model_brake -> vehicle braking
        4. Calculate MSE for both the models and return the weighted combined error.

    Error Handling:
        If training fails for any reason, MSE with huge penalty of 1e6 is returned. 
    
    """
    params = {
        "niterations": trial.suggest_int("niterations", low=50, high=150, step=10),
        "population_size": trial.suggest_int("population_size", low=42, high=60, step = 1),
        "parsimony": trial.suggest_float("parsimony", low=1e-6, high=0.001, log=True),
        "model_selection": trial.suggest_categorical("model_selection", ["best", "accuracy"]),
        "binary_operators": trial.suggest_categorical("binary_operators", [
                                                      ["+","-"],
                                                      ["+","-","*"],
        ]),
        "unary_operators": trial.suggest_categorical("unary_operators",[
                                                     ["sqrt"],
                                                     ["sqrt","sin","cos"],
        ]), 
    }

    # Split data
    X_train, X_val, Y_train, Y_val, mode_train, mode_val= train_test_split(X, Y, mode,
                                                     test_size=0.2, random_state=42)

    # Masks for class
    class_accelerate_train = mode_train == "Accelerate"
    class_brake_train = mode_train == "Brake"
    class_accelerate_val = mode_val == "Accelerate"
    class_brake_val = mode_val == "Brake"

    # Initialize models for both batches
    model_accelerate = PySRRegressor(
        niterations = params["niterations"],
        population_size = params["population_size"],
        parsimony = params["parsimony"],
        model_selection = params["model_selection"],
        binary_operators = params["binary_operators"],
        unary_operators = params["unary_operators"],
        verbosity = 1,
        random_state = 42,
        deterministic = True,
        parallelism="serial", #procs=0 no more valid. If used, Optuna fails to optimize
        ncycles_per_iteration = 50,
    )

    model_brake = PySRRegressor(
        niterations = params["niterations"],
        population_size = params["population_size"],
        parsimony = params["parsimony"],
        model_selection = params["model_selection"],
        binary_operators = params["binary_operators"],
        unary_operators = params["unary_operators"],
        verbosity = 1,
        random_state = 42,
        deterministic = True,
        parallelism="serial", #procs=0 no more valid. If used, Optuna fails to optimize
        ncycles_per_iteration = 50,
    )

    try:
        model_accelerate.fit(X_train[class_accelerate_train, : -1], Y_train[class_accelerate_train])
        model_brake.fit(X_train[class_brake_train, : -1], Y_train[class_brake_train])
        y_pred_accelerate = model_accelerate.predict(X_val[class_accelerate_val, :-1])
        y_pred_brake = model_brake.predict(X_val[class_brake_val, :-1])
        
        val_mse_accelerate  = mean_squared_error(Y_val[class_accelerate_val], y_pred_accelerate)
        val_mse_brake = mean_squared_error(Y_val[class_brake_val], y_pred_brake)

        combined_error = (val_mse_accelerate * len(y_pred_accelerate) + val_mse_brake * len(y_pred_brake)) / len(Y_val)
    except Exception as e:
        print("Error during model training:", str(e))
        combined_error = 1e6 #default val in case of an error
    
    return combined_error
    
# Run Optuna
study = optuna.create_study(storage="sqlite:///vcruise_new.db",
                            direction="minimize")
study.optimize(objective, n_trials=20)

print("Best parameters: ", study.best_params)
print(f"Best validation MSE: {study.best_value: .5f}")