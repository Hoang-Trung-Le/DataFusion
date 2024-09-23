import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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

# Load your data (example for one prediction length)
for pred_length in pred_lengths:
    pred_length = str(pred_length)
    scenario_models[pred_length] = {}

    for scenario_id, config in scenarios.items():
        # Get sampling matrix and parameters for each scenario
        sampling_matrix = config["sampling_matrix"]
        parameters = config["parameters"]

        # Load the features (probabilities of models in partial evaluation)
        X = pd.read_csv(
            f"./fusion/partial/forecast{pred_length}/dsi_ensemble_{int(int(pred_length)/2)}_{pred_length}_ID{scenario_id}_v3.csv",
            index_col="Segment",
        )  # Probabilities of models in partial
        # X["Selected"] = X["Selected"].astype(
        #     "category"
        # )  # Convert "Selected" column to categorical

        # print("X: \n", X)
        # X = X.drop(columns=["Selected"])

        # Load the target (selected model in complete evaluation)
        y = pd.read_csv(
            f"./fusion/complete/forecast{pred_length}/dsi_ensemble_{pred_length}_{pred_length}_ID{scenario_id}_v3.csv",
            index_col="Segment",
        )
        y = y["Selected"].astype("category")  # Convert "Selected" column to categorical
        # Initialize the LabelEncoder
        # Create a single label encoder for both X and y
        shared_label_encoder = LabelEncoder()

        # Concatenate the "Selected" columns from both X and y to ensure the encoder has seen all possible values
        all_selected_values = pd.concat([X["Selected"], y], axis=0).astype(str)
        print("all ", all_selected_values)
        # Fit the shared label encoder on the combined data
        shared_label_encoder.fit(all_selected_values)

        # Apply the encoder to both X and y
        X["Selected_encoded"] = shared_label_encoder.transform(X["Selected"])
        X["Selected_encoded"] = X["Selected_encoded"]
        y_encoded = shared_label_encoder.transform(y)

        # Drop the original "Selected" column in X now that you have the encoded version
        X["Selected_emphasized"] = X[
            "Selected_encoded"
        ]  # Or just duplicate the column multiple times
        X["Selected_dup_1"] = X["Selected_encoded"]
        # X["Selected_dup_2"] = X["Selected_encoded"]
        # X["Selected_dup_3"] = X["Selected_encoded"]
        # X["Selected_dup_4"] = X["Selected_encoded"]
        # X["Selected_dup_5"] = X["Selected_encoded"]

        X = X.drop(columns=["Selected"])
        print("X: \n", X["Selected_encoded"].unique())
        print("X: \n", X)

        print("y:\n", np.unique(y_encoded))

        # Split the data into train and test sets (70% training, 30% testing)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        # Store the split data for each scenario
        scenario_models[pred_length][scenario_id] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "encoder": shared_label_encoder,
        }


# Define the parameter grid for GridSearchCV
param_grid_xgb = {
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200, 300],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2],
}

# Define the parameter grid
param_grid_rf = {
    "n_estimators": [100, 200, 300],  # Number of trees in the forest
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20],  # Maximum depth of the trees
    "min_samples_split": [
        2,
        3,
        4,
        5,
        7,
    ],  # Minimum number of samples required to split a node
    "min_samples_leaf": [
        1,
        2,
        3,
        4,
        5,
    ],  # Minimum number of samples required at each leaf node
    "class_weight": ["balanced", None],  # Handling of class imbalances
    "max_features": ["auto", "sqrt"],
}

# Loop through each scenario and train XGBoost for each one
for pred_length, scenarios in scenario_models.items():

    for scenario_id, data in scenarios.items():
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        print("X train: \n", X_train, "\n y train:\n", y_train)

        # # Convert DataFrame into DMatrix format and enable categorical support
        # dtrain = xgb.DMatrix(X_train, label=y_train.cat.codes, enable_categorical=True)
        # dtest = xgb.DMatrix(X_test, enable_categorical=True)

        # Define an XGBoost model (without any fixed hyperparameters for now)
        xgb_model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=len(np.unique(y_train)),  # Number of unique models
            enable_categorical=True,  # Enable categorical feature support
        )

        # Set up GridSearchCV with 3-fold cross-validation
        grid_search_xgb = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid_xgb,
            scoring="accuracy",  # Use accuracy as the scoring metric
            cv=3,  # Number of folds for cross-validation
            n_jobs=-1,  # Use all available cores
            verbose=1,  # Print the process
        )

        # Initialize the Random Forest Classifier
        rf_classifier = RandomForestClassifier(random_state=42)

        # Initialize GridSearchCV with 5-fold cross-validation
        grid_search_rf = GridSearchCV(
            estimator=rf_classifier,
            param_grid=param_grid_rf,
            cv=5,  # 5-fold cross-validation
            n_jobs=-1,  # Use all available CPU cores
            verbose=1,  # Print the progress of the search
        )

        # Fit and predict with XGBoost
        grid_search_xgb.fit(X_train, y_train)
        best_model_xgb = grid_search_xgb.best_estimator_
        y_pred_proba_xgb = best_model_xgb.predict_proba(X_test)

        # Fit and predict with Random Forest
        grid_search_rf.fit(X_train, y_train)
        best_rf_classifier = grid_search_rf.best_estimator_
        y_pred_proba_rf = best_rf_classifier.predict_proba(X_test)

        # Calculate model accuracies (you'll need to decode the predicted values)
        xgb_accuracy = accuracy_score(y_test, best_model_xgb.predict(X_test))
        rf_accuracy = accuracy_score(y_test, best_rf_classifier.predict(X_test))

        # Normalize accuracies to get weights
        total_accuracy = xgb_accuracy + rf_accuracy
        xgb_weight = xgb_accuracy / total_accuracy
        rf_weight = rf_accuracy / total_accuracy

        # Perform weighted averaging of probabilities
        combined_proba = (xgb_weight * y_pred_proba_xgb) + (rf_weight * y_pred_proba_rf)

        # Make final prediction based on the weighted probabilities
        y_pred_combined = y_train[np.argmax(combined_proba, axis=1)]

        # Evaluate the combined model
        combined_accuracy = accuracy_score(y_test, y_pred_combined)
        print(
            f"Weighted Averaged Accuracy for Scenario {scenario_id}: {combined_accuracy * 100:.2f}%"
        )
        print(f"Classification Report for Scenario {scenario_id} (Weighted Averaging):")
        print(classification_report(y_test, y_pred_combined))

        # # Evaluate accuracy and classification report
        # accuracy = accuracy_score(y_test, y_pred)
        # print(f"Scenario {scenario_id} - XGBoost Accuracy: {accuracy * 100:.2f}%")
        # print(f"Classification Report for Scenario {scenario_id}:")
        # print(classification_report(y_test, y_pred))
