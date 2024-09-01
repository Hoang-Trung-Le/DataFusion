import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DempsterShaferInputHandler:
    def __init__(self, reference_matrix: pd.DataFrame, sampling_matrix: pd.DataFrame):
        """
        Initializes the Dempster-Shafer input handler with reference and sampling matrices.

        Args:
            reference_matrix (pd.DataFrame): The reference matrix.
            sampling_matrix (pd.DataFrame): The sampling matrix.
        """
        if reference_matrix.shape[1] != sampling_matrix.shape[1]:
            raise ValueError(
                "The reference matrix and sampling matrix must have the same number of columns."
            )

        self.reference_matrix = reference_matrix
        self.sampling_matrix = sampling_matrix

        self.normalized_reference_matrix = None
        self.normalized_sampling_matrix = None
        self.scaler = MinMaxScaler()

    def normalize_data(self):
        """
        Normalizes the reference and sampling matrices column-wise using MinMax scaling
        after combining both matrices.
        """
        # Combine the reference and sampling matrices
        combined_matrix = pd.concat(
            [self.reference_matrix, self.sampling_matrix], axis=0
        )

        # Normalize the combined matrix
        normalized_combined_matrix = pd.DataFrame(
            self.scaler.fit_transform(combined_matrix), columns=combined_matrix.columns
        )

        # Split the normalized matrix back into reference and sampling matrices
        self.normalized_reference_matrix = normalized_combined_matrix.iloc[
            : len(self.reference_matrix), :
        ]
        self.normalized_sampling_matrix = normalized_combined_matrix.iloc[
            len(self.reference_matrix) :, :
        ]

    def get_normalized_reference_matrix(self) -> pd.DataFrame:
        """
        Returns the normalized reference matrix.

        Returns:
            pd.DataFrame: The normalized reference matrix.
        """
        if self.normalized_reference_matrix is None:
            raise ValueError(
                "Data has not been normalized. Call `normalize_data()` first."
            )

        return self.normalized_reference_matrix

    def get_normalized_sampling_matrix(self) -> pd.DataFrame:
        """
        Returns the normalized sampling matrix.

        Returns:
            pd.DataFrame: The normalized sampling matrix.
        """
        if self.normalized_sampling_matrix is None:
            raise ValueError(
                "Data has not been normalized. Call `normalize_data()` first."
            )

        return self.normalized_sampling_matrix

    def inverse_normalize(self, normalized_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Inverses the normalization of a given normalized matrix to return to original scale.

        Args:
            normalized_matrix (pd.DataFrame): The normalized matrix.

        Returns:
            pd.DataFrame: The matrix with values restored to the original scale.
        """
        if self.scaler is None:
            raise ValueError(
                "Normalization has not been performed, so inverse normalization cannot be done."
            )

        return pd.DataFrame(
            self.scaler.inverse_transform(normalized_matrix),
            columns=normalized_matrix.columns,
        )


# Example usage:
# Assuming you have two DataFrames: df_reference and df_sampling

# Initialize the handler
# ds_handler = DempsterShaferInputHandler(df_reference, df_sampling)

# Normalize the data
# ds_handler.normalize_data()

# Access the normalized matrices
# normalized_reference = ds_handler.get_normalized_reference_matrix()
# normalized_sampling = ds_handler.get_normalized_sampling_matrix()

# Inverse normalization (if needed)
# original_reference = ds_handler.inverse_normalize(normalized_reference)
