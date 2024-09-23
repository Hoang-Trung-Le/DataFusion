import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
# Example sequence length
# sequence_length = 4  # Define how many time steps in a sequence
# prediction_length = 1

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

# Define the parameter grid
param_grid = {
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
        label_encoder = data["encoder"]
        print("label ", label_encoder.classes_)

        # Initialize the Random Forest Classifier
        rf_classifier = RandomForestClassifier(random_state=42)

        # Initialize GridSearchCV with 5-fold cross-validation
        grid_search = GridSearchCV(
            estimator=rf_classifier,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            n_jobs=-1,  # Use all available CPU cores
            verbose=1,  # Print the progress of the search
        )

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        print(f"Best Hyperparameters for Scenario {scenario_id} - {pred_length}h:")
        print(grid_search.best_params_)

        # Get the best estimator (model) from the grid search
        best_rf_classifier = grid_search.best_estimator_

        # Make predictions on the test set using the best model
        y_pred = best_rf_classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for Scenario {scenario_id} - {pred_length}h: {accuracy}")

        # Print a detailed classification report
        print(f"Classification Report for Scenario {scenario_id} - {pred_length}h:")
        print(
            classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        )
