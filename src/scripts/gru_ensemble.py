import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

def encode_labels(y):
    """
    Encodes categorical labels into integers.

    Args:
        y: List of categorical labels (as Pandas Series or NumPy array).

    Returns:
        Encoded labels and the LabelEncoder instance.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Fit encoder and transform labels
    
    return y_encoded, label_encoder


def create_sequences(X, y, X_indices, y_indices, seq_length, pred_length):
    """
    Creates sequences of data for time series forecasting.

    Args:
        X: Pandas DataFrame containing features (X).
        y: Pandas Series containing targets (y).
        X_indices: Pandas DateTimeIndex corresponding to X.
        y_indices: Pandas DateTimeIndex corresponding to y.
        seq_length: Length of the input sequence.
        pred_length: Length of the output prediction.

    Returns:
        Tuple of lists: (X_seq, y_seq, X_indices_seq, y_indices_seq)
            X_seq: List of input sequences (features).
            y_seq: List of output sequences (targets).
            X_indices_seq: List of indices corresponding to input sequences.
            y_indices_seq: List of indices corresponding to output sequences.
    """
    X_seq, y_seq, X_indices_seq, y_indices_seq = [], [], [], []
    
    for i in range(len(X) - seq_length - pred_length + 1):
        # Input sequence (all columns of X)
        X_seq.append(X.iloc[i : i + seq_length].values)
        
        # Output target sequence (selected columns of y)
        y_seq.append(y[i + seq_length : i + seq_length + pred_length])
        
        # Indices for X and y
        X_indices_seq.append(X_indices[i : i + seq_length])
        y_indices_seq.append(y_indices[i + seq_length : i + seq_length + pred_length])
    
    return np.array(X_seq), np.array(y_seq), X_indices_seq, y_indices_seq


def train_test_split_ts(X, y, split_ratio=0.8):
    """
    Splits time series data into training and testing sets.

    Args:
      X: Time series features.
      y: Time series target variable.
      split_ratio: Proportion of data to use for training.

    Returns:
      X_train, X_test, y_train, y_test
    """
    split_index = int(len(X) * split_ratio)
    
    # Split the data
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test


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
sequence_length = 3  # Define how many time steps in a sequence
prediction_length = 1

# Load your data (example for one prediction length)
for pred_length in pred_lengths:
    scenario_models[str(pred_length)] = {}

    for scenario_id, config in scenarios.items():
        # Load the features (probabilities of models in partial evaluation)
        X = pd.read_csv(
            f"./fusion/partial/forecast{pred_length}/dsi_ensemble_{int(pred_length/2)}_{pred_length}_ID{scenario_id}_v3.csv",
            index_col="Segment"
        )
        # X["Selected"] = X["Selected"].astype("category")
        X= X.drop(columns = ["Selected"])


        # Load the target (selected model in complete evaluation)
        y = pd.read_csv(
            f"./fusion/complete/forecast{pred_length}/dsi_ensemble_{pred_length}_{pred_length}_ID{scenario_id}_v3.csv",
            index_col="Segment"
        )
        y = y["Selected"].astype("category")

        # Encode the labels (for y)
        y_encoded, label_encoder = encode_labels(y)

        # Create sequences for both X and y
        X_seq, y_seq, X_indices_seq, y_indices_seq = create_sequences(
            X, y_encoded, X.index, y.index, seq_length, prediction_length
        )

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split_ts(X_seq, y_seq, split_ratio=0.8)

        # Store the data for each scenario
        scenario_models[str(pred_length)][scenario_id] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "label_encoder": label_encoder  # Save the encoder to decode labels later
        }



scenario_predictions = {}
for pred_length, scenarios in scenario_models.items():
    scenario_predictions[pred_length] = {}

    for scenario_id, data in scenarios.items():
        print(f"Training model for scenario {scenario_id} and prediction length {pred_length}...")

        # Prepare train and test data for the scenario
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

        gru_model = Sequential([
            GRU(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
            Dense(64, activation='relu'),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])

        gru_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        gru_model.summary()

        # Train the model
        gru_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)


        # Predict with LSTM/GRU/XGBoost
        y_pred = gru_model.predict(X_test)  # Or use GRU or XGBoost model

        # Convert predictions to class labels (if using softmax)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Compare predicted labels with true labels (y_test_encoded)
        print(y_pred_classes)
        print("len classes",len(y_pred_classes))
        accuracy = np.mean(y_pred_classes == y_test)

        print(f"Scenario {scenario_id}, Prediction Length {pred_length}, Accuracy: {accuracy}")
