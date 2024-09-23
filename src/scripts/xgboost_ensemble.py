import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

seq_length = 12
pred_lengths = [6, 12]
model_names = ["1D-CNN", "ANN", "BiLSTM", "LSTM", "CNN-LSTM", "GRU"]

# Define scenarios
scenarios = {
    1: {"sampling_matrix": np.array([[0, 1]]), "parameters": ["RMSE", "Pearson_r"]},
    2: {
        "sampling_matrix": np.array([[0, 1, 1]]),
        "parameters": ["RMSE", "Pearson_r", "R2"],
    },
    3: {
        "sampling_matrix": np.array([[0, 0, 1]]),
        "parameters": ["RMSE", "MAE", "Pearson_r"],
    },
    4: {"sampling_matrix": np.array([[0, 1]]), "parameters": ["RMSE", "R2"]},
    5: {"sampling_matrix": np.array([[0, 0, 1]]), "parameters": ["RMSE", "MAE", "R2"]},
    6: {
        "sampling_matrix": np.array([[0, 0, 1, 1]]),
        "parameters": ["RMSE", "MAE", "Pearson_r", "R2"],
    },
}

model_id = 3
scenario_models = {}

for pred_length in pred_lengths:
    pred_length = str(pred_length)
    scenario_models[pred_length] = {}

    for scenario_id, config in scenarios.items():
        # Get sampling matrix and parameters for each scenario
        sampling_matrix = config["sampling_matrix"]
        parameters = config["parameters"]

        # Load the features (probabilities of models in partial evaluation)
        X = pd.read_csv(
            f"./fusion/partial/forecast{pred_length}/dsi_ensemble_{int(int(pred_length)/2)}_{pred_length}_ID{scenario_id}.csv",
            index_col="Segment",
        )  # Probabilities of models in partial
        X["Selected"] = X["Selected"].astype(
            "category"
        )  # Convert "Selected" column to categorical
        print("X: \n", X)

        # Load the target (selected model in complete evaluation)
        y = pd.read_csv(
            f"./fusion/complete/forecast{pred_length}/dsi_ensemble_{pred_length}_{pred_length}_ID{scenario_id}.csv",
            index_col="Segment",
        )
        y = y["Selected"].astype("category")  # Convert "Selected" column to categorical
        print("y:\n", y)

        # Split the data into train and test sets (70% training, 30% testing)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Store the split data for each scenario
        scenario_models[pred_length][scenario_id] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }


# Define the parameter grid for GridSearchCV
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200, 300],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2],
}

# Loop through each scenario and train XGBoost for each one
for pred_length, scenarios in scenario_models.items():

    for scenario_id, data in scenarios.items():
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

        # Convert DataFrame into DMatrix format and enable categorical support
        dtrain = xgb.DMatrix(X_train, label=y_train.cat.codes, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, enable_categorical=True)

        # Define an XGBoost model (without any fixed hyperparameters for now)
        xgb_model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=len(y_train.cat.categories),  # Number of unique models
            enable_categorical=True,  # Enable categorical feature support
        )

        # Set up GridSearchCV with 3-fold cross-validation
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring="accuracy",  # Use accuracy as the scoring metric
            cv=3,  # Number of folds for cross-validation
            n_jobs=-1,  # Use all available cores
            verbose=1,  # Print the process
        )

        # Fit GridSearchCV to the training data
        grid_search.fit(X_train, y_train.cat.codes)
        # grid_search.fit(dtrain, y_train.cat.codes)

        # Get the best parameters and the best model
        best_model = grid_search.best_estimator_
        print(f"Best parameters for Scenario {scenario_id}: {grid_search.best_params_}")

        # Predict on the test set with the best model
        y_pred_codes = best_model.predict(X_test)
        # y_pred_codes = best_model.predict(dtest)

        # Decode the predictions back to the original model names
        y_pred = y_train.cat.categories[y_pred_codes]

        # Evaluate accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Scenario {scenario_id} - XGBoost Accuracy: {accuracy * 100:.2f}%")
        print(f"Classification Report for Scenario {scenario_id}:")
        print(classification_report(y_test, y_pred))
