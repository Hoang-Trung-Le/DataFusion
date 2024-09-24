import numpy as np
import pandas as pd
import xgboost as xgb
import cupy as cp
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    max_error,
    mean_absolute_percentage_error,
)
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore", category=UserWarning)


def calculate_statistics(y_true, y_pred):
    """Calculates statistical metrics for model evaluation."""
    # Ensure both arrays are flattened to 1D
    # y_true = y_true.to_numpy().flatten()
    # y_pred = y_pred.flatten()

    # rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    max_error_value = max_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mae, pearson_r, r2, max_error_value, mape


def evaluate_forecasts(forecasts_df, y_test_df):
    """Evaluate forecasts row-by-row and return statistics."""
    statistics = []

    for segment in forecasts_df.index:
        y_true = y_test_df.loc[segment].to_numpy().flatten()
        y_pred = forecasts_df.loc[segment].to_numpy().flatten()

        rmse, mae, pearson_r, r2, max_error_value, mape = np.round(
            calculate_statistics(y_true, y_pred), 6
        )

        statistics.append(
            {
                "Segment": segment,
                "RMSE": rmse,
                "MAE": mae,
                "Pearson_r": pearson_r,
                "R2": r2,
                "Max_Error": max_error_value,
                "MAPE": mape,
            }
        )

    # Convert the statistics list to a DataFrame
    statistics_df = pd.DataFrame(statistics)

    return statistics_df


def create_sequences_ser(data, seq_length):
    """
    Creates sequences of data for time series forecasting.

    Args:
        data: Pandas DataFrame containing time series data.
        data_indices: Pandas DateTimeIndex corresponding to the data.
        seq_length: Length of the input sequence.
        pred_length: Length of the output prediction.

    Returns:
        Tuple of lists: (X, y, X_indices, y_indices)
            X: List of input sequences.
            y: List of output targets.
            X_indices: List of indices corresponding to the input sequences.
            y_indices: List of indices corresponding to the output targets.
    """
    # X, y, X_indices, y_indices = [], [], [], []
    X, X_indices = [], []
    data_indices = data.index.values
    for i in range(len(data) - seq_length + 1):
        # Extract input sequence (all columns) and target values (LIVERPOOL_PM2.5)
        X.append(
            data.iloc[i : (i + seq_length)].values
        )  # Use all columns for input sequences
        # y.append(
        #     data.iloc[(i + seq_length) : (i + seq_length + pred_length)].values
        # )  # Use LIVERPOOL_PM2.5 for targets
        X_indices.append(data_indices[i : (i + seq_length)])
        # y_indices.append(
        #     data_indices[(i + seq_length) : (i + seq_length + pred_length)]
        # )
    return np.array(X), X_indices


def create_lag_features(df, column, num_lags):
    """
    Creates lag features for a specific column.

    Parameters:
    - df: DataFrame containing the original data.
    - column: Column name for which to create lag features.
    - num_lags: Number of lag features to create.

    Returns:
    - DataFrame with new lag features.
    """
    for lag in range(1, num_lags + 1):
        df[f"lag_{column}_{lag}"] = df[column].shift(lag)
    return df


def create_target_columns(df, column, forecast_horizons):
    """
    Creates target columns for multi-step ahead forecasts.

    Parameters:
    - df: DataFrame containing the original data.
    - column: Column name for which to create target columns.
    - forecast_horizons: List of forecast steps (e.g., [6, 12] for 6-hour and 12-hour predictions).

    Returns:
    - DataFrame with target columns.
    """
    for horizon in forecast_horizons:
        df[f"target_{column}_{horizon}h"] = df[column].shift(-horizon)
    return df


def split_train_test(df, features, targets, train_size=0.9):
    """
    Splits data into train and test sets.

    Parameters:
    - df: DataFrame with all features and targets.
    - features: List of feature column names.
    - targets: List of target column names.
    - train_size: Proportion of data to use for training.

    Returns:
    - X_train, X_test, y_train, y_test
    """
    n_train = int(len(df) * train_size)

    X_train = df[features].iloc[:n_train]
    X_test = df[features].iloc[n_train:]

    y_train = df[targets].iloc[:n_train]
    y_test = df[targets].iloc[n_train:]

    return X_train, X_test, y_train, y_test


def save_test_data(df, sequence, dataset_type, output_data_file, output_indices_file):
    """
    Saves test data and their corresponding timestamps/indices.

    Parameters:
    - df: DataFrame containing test data.
    - target_columns: List of target column names.
    - indices: Datetime index of the test data.
    - output_data_file: Filename to save the test data.
    - output_indices_file: Filename to save the test indices.

    Returns:
    - None
    """
    # Saving the data
    label = ""
    if dataset_type == "test":
        label = "OBS"
    elif dataset_type == "forecast":
        label = "F"

    test_data, test_data_indices = create_sequences_ser(df, sequence)
    print("test data:\n", test_data)
    test_data_df = pd.DataFrame(
        test_data,
        index=[f"S{i+1}" for i in range(test_data.shape[0])],
        columns=[f"{label}{j+1}" for j in range(test_data.shape[1])],
    )
    test_data_df.index.name = "Segment"
    test_data_df.to_csv(output_data_file, index=True, header=True)

    # Saving the indices
    indices_df = pd.DataFrame(
        test_data_indices,
        index=[f"S{i+1}" for i in range(test_data.shape[0])],
        columns=[f"{label}{j+1}" for j in range(test_data.shape[1])],
    )
    indices_df.index.name = "Segment"
    indices_df.to_csv(output_indices_file, index=True, header=True)


def perform_grid_search(X_train, y_train):
    """
    Performs grid search for XGBRegressor.

    Parameters:
    - X_train: Training feature set.
    - y_train: Training target set.

    Returns:
    - Best estimator from grid search.
    """
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],  # Explore learning rate
        "n_estimators": [100, 200, 300],  # Number of trees
        "max_depth": [3, 5, 7],  # Tree depth
        "min_child_weight": [1, 5, 10],  # Minimum sum of instance weight (hessian)
        "gamma": [0, 0.1, 0.2],  # Minimum loss reduction for split
        "subsample": [0.6, 0.8, 1.0],  # Fraction of samples for each tree
        "colsample_bytree": [0.6, 0.8, 1.0],  # Fraction of features for each tree
        # "reg_alpha": [0, 0.01, 0.1, 1],  # L1 regularization
        # "reg_lambda": [1, 10, 100],  # L2 regularization
    }
    xgbr = xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist", device="cuda"
    )

    grid_search = GridSearchCV(
        xgbr, param_grid, cv=3, scoring="neg_mean_squared_error", verbose=1
    )

    # Convert Numpy arrays to CuPy arrays
    # X_train = cp.array(X_train)  # Move your X_train to GPU
    # y_train = cp.array(y_train)  # Move your y_train to GPU
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


df = pd.read_csv(
    "./data/processed/LIVERPOOL_PM2.5_hourly.csv",
    index_col="datetime",
    parse_dates=True,
)
df = df[["LIVERPOOL_PM2.5"]]

# Print the dummy DataFrame
print(df)
seq_length = 12
# Step 1: Create lag features
df = create_lag_features(df, "LIVERPOOL_PM2.5", num_lags=seq_length)
print("df lag:\n", df)
# Step 2: Create target columns for 6 and 12 hours ahead
df = create_target_columns(df, "LIVERPOOL_PM2.5", forecast_horizons=[6, 12])

# Step 3: Drop rows with NaN values due to lagging
df = df.dropna()


targets = {6: "target_LIVERPOOL_PM2.5_6h", 12: "target_LIVERPOOL_PM2.5_12h"}
best_models = {}
X_train, X_test, y_train, y_test, y_pred = {}, {}, {}, {}, {}

for pred_horizon, target_col in targets.items():
    # Step 4: Split the data into train/test sets
    (
        X_train[pred_horizon],
        X_test[pred_horizon],
        y_train[pred_horizon],
        y_test[pred_horizon],
    ) = split_train_test(
        df,
        features=[f"lag_LIVERPOOL_PM2.5_{i}" for i in range(1, 13)],
        targets=[f"target_LIVERPOOL_PM2.5_{pred_horizon}h"],
    )
    print("X test:\n", X_test)

    save_test_data(
        y_test[pred_horizon][target_col],
        pred_horizon,
        "test",
        f"./forecasts_ensemble/test/test_data_forecast{pred_horizon}.csv",
        f"./forecasts_ensemble/test/test_indices_forecast{pred_horizon}.csv",
    )

    # Step 5: Perform grid search for best model
    # best_models[pred_horizon] = perform_grid_search(
    #     X_train[pred_horizon], y_train[pred_horizon][target_col]
    # )
    best_models[pred_horizon] = perform_grid_search(
        cp.array(X_train[pred_horizon]), cp.array(y_train[pred_horizon][target_col])
    )

    # Step 6: Make predictions
    # y_pred[pred_horizon] = best_models[pred_horizon].predict(
    #     X_test[pred_horizon]
    # )
    y_pred[pred_horizon] = best_models[pred_horizon].predict(
        cp.array(X_test[pred_horizon])
    )
    print("y pred:\n", y_pred)
    print("len pred:", len(y_pred[pred_horizon]))
    print("len X test:", len(X_test[pred_horizon].index))
    y_pred_df = pd.DataFrame(
        y_pred[pred_horizon], index=X_test[pred_horizon].index, columns=[target_col]
    )
    print("y pred df:\n", y_pred_df)
    save_test_data(
        y_pred_df[target_col],
        pred_horizon,
        "forecast",
        f"./forecasts_ensemble/forecast{pred_horizon}/LIVERPOOL_forecast_{seq_length}_{pred_horizon}_XGB.csv",
        f"forecast_indices_{pred_horizon}h.csv",
    )
    # Step 7: Evaluate the predictions
    forecasts_df = pd.read_csv(
        f"./forecasts_ensemble/forecast{pred_horizon}/LIVERPOOL_forecast_{seq_length}_{pred_horizon}_XGB.csv",
        index_col="Segment",
    )
    observations_df = pd.read_csv(
        f"./forecasts_ensemble/test/test_data_forecast{pred_horizon}.csv",
        index_col="Segment",
    )
    evaluation_results = evaluate_forecasts(forecasts_df, observations_df)
    evaluation_results.to_csv(
        f"./forecasts_ensemble/stats/complete/forecast{pred_horizon}/LIVERPOOL_stats_{seq_length}_{pred_horizon}_XGB.csv",
        index=False,
    )
    # evaluation_results_df = pd.DataFrame(evaluation_results, index=[pred_horizon])
    print(evaluation_results)
    print(best_models[pred_horizon])
