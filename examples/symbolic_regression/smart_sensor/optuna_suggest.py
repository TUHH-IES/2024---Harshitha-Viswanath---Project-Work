"""
optuna_suggest.py

This script performs hyperparameter optimization for SR 
using Optuna and PySRRegressor to model the output of a 
smart sensor CPS. 
 
Workflow:
    1. Load sampled data from "sampled_data.csv".
    2. Split dataset into training and validation sets.
    3. Use Optuna to optimize hyperparameters for two seperate batches:
        - class_compress_train and class_compress_val -> Smart sensor in compression mode = 1
        - class_amplify_train and class_amplify_val -> Smart sensor in amplification mode = 2
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
X = df.drop(["time", "output_y"], axis=1).values
Y = df["output_y"].values
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
            - model_compress -> compression mode of Smart Sensor
            - model_amplify -> amplification mode of Smart sensor
        4. Calculate MSE for both the models and return the weighted combined error.

    Error Handling:
        If training fails for any reason, MSE with huge penalty of 1e6 is returned. 
    
    """
    params = {
        "niterations": trial.suggest_int("niterations", low=50, high=150, step=10),
        "populations": trial.suggest_int("populations", low=42, high=60, step = 1),
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
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Masks for class
    class_compress_train = X_train[:, class_id] == 1
    class_amplify_train = X_train[:, class_id] == 2
    class_compress_val = X_val[:, class_id] == 1
    class_amplify_val = X_val[:, class_id] == 2

    # Initialize models for both batches
    model_compress = PySRRegressor(
        niterations = params["niterations"],
        populations = params["populations"],
        parsimony = params["parsimony"],
        model_selection = params["model_selection"],
        binary_operators = params["binary_operators"],
        unary_operators = params["unary_operators"],
        verbosity = 1,
        random_state = 42,
        deterministic = True,
        parallelism = "serial",    #procs=0 no more valid. If used, Optuna fails to optimize
        ncycles_per_iteration = 50,
    )

    model_amplify = PySRRegressor(
        niterations = params["niterations"],
        populations = params["populations"],
        parsimony = params["parsimony"],
        model_selection = params["model_selection"],
        binary_operators = params["binary_operators"],
        unary_operators = params["unary_operators"],
        verbosity = 1,
        random_state = 42,
        deterministic = True,
        parallelism = "serial",     #procs=0 no more valid. If used, Optuna fails to optimize
        ncycles_per_iteration = 50,
    )

    try:
        model_compress.fit(X_train[class_compress_train, : -1], Y_train[class_compress_train])
        model_amplify.fit(X_train[class_amplify_train, : -1], Y_train[class_amplify_train])
        y_pred_compress = model_compress.predict(X_val[class_compress_val, :-1])
        y_pred_amplify = model_amplify.predict(X_val[class_amplify_val, :-1])
        
        val_mse_compress  = mean_squared_error(Y_val[class_compress_val], y_pred_compress)
        val_mse_amplify = mean_squared_error(Y_val[class_amplify_val], y_pred_amplify)

        combined_error = (val_mse_compress * len(y_pred_compress) + val_mse_amplify * len(y_pred_amplify)) / len(Y_val)
    except Exception:
        combined_error = 1e6 #default val in case of an error
    
    return combined_error


# Run Optuna
study = optuna.create_study(storage="sqlite:///smartsensor.db",
                            direction="minimize")
study.optimize(objective, n_trials=20)

print("Best parameters: ", study.best_params)
print(f"Best validation MSE: {study.best_value: .5f}")