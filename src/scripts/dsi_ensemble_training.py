import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    GRU,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Bidirectional,
    TimeDistributed,
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    max_error,
    mean_absolute_percentage_error,
)
from scipy.stats import pearsonr
from collections import OrderedDict
import matplotlib.pyplot as plt

"""# Model

## Models
"""


# Function to create models
def create_models(input_shape, output_length):
    models = {}

    # 1D CNN
    cnn_model = Sequential(
        [
            Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(output_length),
        ]
    )
    models["1D CNN"] = cnn_model

    # LSTM
    lstm_model = Sequential(
        [
            LSTM(
                128, activation="tanh", return_sequences=True, input_shape=input_shape
            ),
            LSTM(128, activation="tanh"),
            Dense(64, activation="relu"),
            Dense(output_length),
        ]
    )
    models["LSTM"] = lstm_model

    # # GRU
    # gru_model = Sequential(
    #     [
    #         GRU(128, activation="tanh", input_shape=input_shape),
    #         Dense(64, activation="relu"),
    #         Dense(output_length),
    #     ]
    # )
    # models["GRU"] = gru_model

    # # BiLSTM
    # bilstm_model = Sequential(
    #     [
    #         Bidirectional(LSTM(128, activation="tanh"), input_shape=input_shape),
    #         Dense(64, activation="relu"),
    #         Dense(output_length),
    #     ]
    # )
    # models["BiLSTM"] = bilstm_model

    # # CNN-LSTM
    # cnn_lstm_model = Sequential(
    #     [
    #         Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape),
    #         # MaxPooling1D(pool_size=2),
    #         # Reshape((input_shape)),
    #         LSTM(100, activation="tanh"),
    #         Dense(100, activation="relu"),
    #         Dense(output_length),
    #     ]
    # )
    # # print(cnn_lstm_model.summary())
    # models["CNN-LSTM"] = cnn_lstm_model

    # # ANN
    # ann_model = Sequential(
    #     [
    #         Flatten(input_shape=input_shape),
    #         Dense(100, activation="relu"),
    #         Dense(100, activation="relu"),
    #         Dense(output_length),
    #     ]
    # )
    # models["ANN"] = ann_model

    return models


"""## Training functions"""


# Training and evaluation function
def train_and_evaluate(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs,
    batch_size,
    patience,
    forecast_hours,
    seq_length,
):
    history = {}
    results = {}
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )  # Configure early stopping

    for name, model in models.items():
        model.compile(
            optimizer="adam", loss="mse", metrics=["mse"]
        )  # Add MSE as a metric

        # Add a check to ensure input and output dimensions match
        if y_train.shape[1] != model.output_shape[-1]:
            raise ValueError(
                f"Output dimension mismatch for model {name}. "
                f"Expected: {y_train.shape[1]}, Got: {model.output_shape[-1]}"
            )

        history[name] = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],  # Add the early stopping callback
            verbose=1,
        )
        results[name] = model.evaluate(X_test, y_test, verbose=1)
        # print(f'{name} Test Loss: {results[name][0]}, Test MSE: {results[name][1]}')  # Print both loss and MSE

        # Save the model
        model_filename = (
            f"LIVERPOOL_forecast_PM2.5_{seq_length}_{forecast_hours}_{name}.h5"
        )
        save_dir = "./models"
        model.save(os.path.join(save_dir, model_filename))
        print(
            f"Model {name} for {forecast_hours}-hour forecast saved as {model_filename}"
        )

        print(f"Test MSE: {results[name][1]}")  # Print both loss and MSE
        plot_training_history(
            history[name], name, forecast_hours, key
        )  # Plot training history for each model
    return history, results


"""## Forecast functions"""


# Forecasting function
def forecast(models, X, pred_length, seq_length):
    forecasts = {}
    for name, model in models.items():
        forecast = model.predict(X[-1].reshape(1, seq_length, X.shape[2]), verbose=1)
        # print(np.array(forecast).shape)
        # forecasts[name] = scaler.inverse_transform(forecast).flatten()
        # forecasts[name] = scaler_target.inverse_transform(forecast).flatten()

        forecasts[name] = forecast
    return forecasts


def forecasts(models, X, pred_length, seq_length):
    forecasts = {name: [] for name in models.keys()}
    for i in range(len(X) - seq_length):
        input_sequence = X[i].reshape((1, seq_length, 1))
        # print("input sequene: ", input_sequence)
        # print("input sequene: ", input_sequence.shape)
        if i >= 20:
            break
        for name, model in models.items():
            # print("reshaped input: ", input_sequence)
            print(f"Forecating {i} segment over {len(X)} for {name} for {pred_length}-hour")
            print("reshaped input: ", input_sequence.shape)
            print("prediction process: ", 100*i/len(X))
            forecast = model.predict(
                input_sequence,
                verbose=1,
            )
            forecasts[name].append(
                forecast.flatten()
            )  # Flatten the forecast to ensure consistent shape
    return {name: np.array(forecast) for name, forecast in forecasts.items()}


"""## Evaluation functions"""


def calculate_statistics(y_true, y_pred):
    """Calculates statistical metrics for model evaluation."""
    # Ensure both arrays are flattened to 1D
    y_true = y_true.to_numpy().flatten()
    y_pred = y_pred.flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    max_error_value = max_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mae, pearson_r, r2, max_error_value, mape


def evaluate_forecasts(forecasts, y_true):
    statistics = {}
    for name, forecast in forecasts.items():
        print("True: ", y_true.shape)
        print("Forecast: ", forecast.shape)
        print("True: ", y_true.to_numpy)
        print("Forecast: ", forecast[0])
        rmse, mae, pearson_r, r2, max_error_value, mape = np.round(
            calculate_statistics(y_true, forecast[0]), 6
        )
        statistics[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "Pearson's r": pearson_r,
            "R²": r2,
            "Max Error": max_error_value,
            "MAPE": mape,
        }
        # print(f"{name} - RMSE: {rmse}, MAE: {mae}, Pearson's r: {pearson_r}, R²: {r2}")
    return statistics


# Convert statistics to DataFrame and display
def statistics_to_dataframe(statistics, pred_length):
    df = pd.DataFrame(statistics).T
    df.index.name = "Model"
    df.columns = ["RMSE", "MAE", "Pearson's r", "R²", "Max Error", "MAPE"]
    df["Pred. Length"] = pred_length
    return df


"""## Plotting functions"""


## Ploting historical training and validating
def plot_training_history(history, name, forecast_hours, key):
    plt.figure(figsize=(12, 3))
    # plt.plot(history.history['loss'], label='Train Loss')
    # plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history["mse"], label="Train MSE")  # Plot MSE
    plt.plot(history.history["val_mse"], label="Val MSE")  # Plot MSE
    plt.title(
        f"Training vs. Validation Metrics for {name} at {forecast_hours}-hour forecast [{key}]"
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    # plt.show()


def plot_forecasts(data, forecasts, pred_length, title):
    # print(data)
    # print(forecasts)
    plt.figure(figsize=(15, 4))
    plt.plot(
        data.index[-(seq_length + pred_length) :],
        data.values[-(seq_length + pred_length) :],
        label="Real_OBS",
        linewidth=3,
    )
    for name, forecast in forecasts.items():
        # print(name)
        # print(forecast.shape)
        # print(data.index[-forecast.shape[1]:])
        forecast = forecast.flatten()
        print("forecast", forecast)
        index = data.index[-forecast.shape[0] :]
        print("index", index)
        plt.plot(index, forecast, markersize=10, label=f"{name} Forecast")

    # plt.plot(data.index[-len(forecasts['LSTM']):], forecasts['LSTM'], marker='o', markersize=10,  label=f'Forecast', color = 'red')
    plt.xticks(rotation=0)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.show()


"""# Load data

### Station
"""

data = {}
data["station"] = pd.read_csv(
    "./data/processed/LIVERPOOL_PM2.5_hourly.csv",
    index_col="datetime",
    parse_dates=True,
)
data["station"].index = pd.to_datetime(data["station"].index, format="%d/%m%Y %H:%M")
data["station"].index = data["station"].index.strftime("%Y-%m-%d %H:%M:%S")
# data['station'].head()

# Example usage:

# Assuming 'data' is a DataFrame with your time series data
window_sizes = [12, 24]  # 12-hour and 24-hour windows (assuming hourly data)


"""# Data windowing"""

seq_length = 12  ## Number of historical input values
pred_lengths = [6, 12]  ## Number of future out values


def create_sequences_ser(data, data_indices, seq_length, pred_length):
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
    X, y, X_indices, y_indices = [], [], [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        # Extract input sequence (all columns) and target values (PM2.5_DSI)
        X.append(
            data.iloc[i : (i + seq_length)].values
        )  # Use all columns for input sequences
        y.append(
            data.iloc[(i + seq_length) : (i + seq_length + pred_length)].values
        )  # Use PM2.5_DSI for targets
        X_indices.append(data_indices[i : (i + seq_length)])
        y_indices.append(
            data_indices[(i + seq_length) : (i + seq_length + pred_length)]
        )
    return np.array(X), np.array(y), X_indices, y_indices


start_date = "2018-01-01 01:00"
end_date = "2023-10-01 00:00"

liv_data = data["station"][start_date:end_date]
# liv_data_indices = data['station'].index

X = dict()
y = dict()
X_indices = dict()
y_indices = dict()

for i in pred_lengths:
    X[f"{i}"], y[f"{i}"], X_indices[f"{i}"], y_indices[f"{i}"] = create_sequences_ser(
        liv_data, liv_data.index, seq_length, i
    )


"""# Split data"""

# Define the split ratio
split_ratio = 0.9


def train_test_split_ts(X, y, split_ratio=0.8):
    """
    Splits time series data into training and testing sets.

    Args:
      X: Time series features.
      y: Time series target variable.
      split_ratio: Proportion of data to use for testing.

    Returns:
      X_train, X_test, y_train, y_test
    """
    X_train = dict()
    y_train = dict()
    X_test = dict()
    y_test = dict()

    split_index = int(len(X) * split_ratio)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


### Re arrange the data following the order of forecast
X = OrderedDict(sorted(X.items()))
y = OrderedDict(sorted(y.items()))
X_indices = OrderedDict(sorted(X_indices.items()))
y_indices = OrderedDict(sorted(y_indices.items()))


X_train = dict()
y_train = dict()
X_test = dict()
y_test = dict()

for key in X.keys():
    X_train[key], X_test[key], y_train[key], y_test[key] = train_test_split_ts(
        X[key], y[key], split_ratio
    )
    # print(X_train[key].shape, X_test[key].shape, y_train[key].shape, y_test[key].shape)
    # print(X_test)

"""# Normalize data"""

scaler = dict()
for key in X_train.keys():
    scaler[key] = MinMaxScaler()
# Initialize dictionaries to hold the scaled data
X_train_scaled = dict()
X_test_scaled = dict()
y_train_scaled = dict()
y_test_scaled = dict()

# Fit and transform the training data, then transform the test data
for key in X_train.keys():
    # Fit the scaler on training data
    X_train_scaled[key] = (
        scaler[key]
        .fit_transform(X_train[key].reshape(-1, X_train[key].shape[-1]))
        .reshape(X_train[key].shape)
    )

    # Transform the test data
    X_test_scaled[key] = (
        scaler[key]
        .transform(X_test[key].reshape(-1, X_test[key].shape[-1]))
        .reshape(X_test[key].shape)
    )

    # Fit the scaler on training target data
    y_train_scaled[key] = (
        scaler[key]
        .fit_transform(y_train[key].reshape(-1, y_train[key].shape[-1]))
        .reshape(y_train[key].shape)
    )

    # Transform the test target data
    y_test_scaled[key] = (
        scaler[key]
        .transform(y_test[key].reshape(-1, y_test[key].shape[-1]))
        .reshape(y_test[key].shape)
    )

    # print(X_train_scaled[key].shape, X_test_scaled[key].shape, y_train_scaled[key].shape, y_test_scaled[key].shape)
    # print("-"*3 + f" {key} " + "-"*3)
    # print(X_train_scaled)

"""# Train model"""

epochs = 50
batch_size = 512
patience = 5

models = dict()
for forecast_length in pred_lengths:
    for i, key in enumerate(X_train.keys()):
        if (
            str(forecast_length)
        ) in key:  ### align length of forecast with segments of data
            input_shape = (seq_length, X_train_scaled[key].shape[-1])
            models[key] = create_models(input_shape, forecast_length)
            history, results = train_and_evaluate(
                models[key],
                X_train_scaled[key],
                y_train_scaled[key],
                X_test_scaled[key],
                y_test_scaled[key],
                epochs,
                batch_size,
                patience,
                forecast_length,
                seq_length,
            )


"""# Forecast"""

forecast_results_scaled = {}
for i, key in enumerate(X_train.keys()):
    ## Make forecast and plot
    forecast_results_scaled[key] = forecasts(
        models[key], X_test_scaled[key], key, seq_length
    )

print(forecast_results_scaled)

"""## Inverse forecast scaling"""

# prompt: inverse the scaling of the forecast result from the variable forecast_resutls_scaled from the scaler scaler_X

forecast_results = {}
for key in forecast_results_scaled.keys():
    forecast_results[key] = {}
    for model_name, forecast in forecast_results_scaled[key].items():
        # Print to check shapes before transformation
        print(f"Forecast shape before inverse transform: {forecast.shape}")

        # Reshape forecast to fit scaler's expected input
        # Assuming the original data had 6 features, reshape to (num_samples, 6)
        num_samples = forecast.shape[0]
        forecast_reshaped = forecast.reshape(
            num_samples, -1
        )  # -1 infers the number of features (6 in this case)

        # Apply the inverse transform
        forecast_inverse_transformed = scaler[key].inverse_transform(forecast_reshaped)

        # Reshape back to the original forecast shape if needed
        # forecast_original_shape = forecast_inverse_transformed.reshape(forecast.shape)

        # Flatten the forecast result to match the desired output format
        forecast_results[key][model_name] = forecast_inverse_transformed
print(forecast_results)

"""## Plot forecast"""

forecast_results


pred_lengths = [6, 12]
for forecast_length in pred_lengths:

    plot_forecasts(
        liv_data,
        forecast_results[f"{forecast_length}"],
        forecast_length,
        f"{forecast_length} Hours Forecast",
    )
    ## Forecast stats
    statistics = evaluate_forecasts(
        forecast_results[f"{forecast_length}"], liv_data.iloc[-(forecast_length):]
    )
    ## convert to dataframes
    df_stat = statistics_to_dataframe(statistics, forecast_length)
    print(f"---- > Statistic of forecast for {forecast_length}-hour: \n", df_stat)
    print("-" * 50)
    print("\n")
