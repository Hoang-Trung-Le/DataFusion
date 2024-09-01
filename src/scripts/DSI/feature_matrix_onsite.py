import os
import numpy as np
import pandas as pd
from datetime import datetime
from utils.data_process.process_dict import mirror_and_populate_dict
from utils.data_process.time_functions import convert_to_unix


class FeatureMatrixOnsite:
    def __init__(self, start, end, is_auto_mapping=False):
        # self.hypotheses_info = hypotheses_info
        self.is_auto = is_auto_mapping
        self.start_time = start
        self.end_time = end
        # self.temp_features_matrix_dict = {}
        self.features_matrix_dict = {}
        # self.features_matrix_np = []
        self.feature_df = None
        self.read_data_from_files()
        self.assign_feature_matrices()

    def read_data_from_files(self):
        """
        This function reads the data from the files
        """
        feature_matrix_file = (
            "../data/aqms/normalized/aqms_1141_corrected_normalized.csv"
        )
        feature_columns = [
            "HUMID",
            "TEMP",
            "PM2.5",
            "PM10",
        ]  # Specify the columns you want to extract
        self.feature_df = pd.read_csv(
            feature_matrix_file,
            parse_dates=["datetime_utc"],
            index_col="datetime_utc",
        )
        self.feature_df = self.feature_df[feature_columns][
            (self.feature_df["timestamp"] >= convert_to_unix(self.start_time))
            & (self.feature_df["timestamp"] <= convert_to_unix(self.end_time))
        ]
        print(self.feature_df)

    def assign_feature_matrices(self):
        """
        This function assigns the feature matrices to the hypotheses
        """
        # Convert the 'timestamp' column in feature_df to datetimeindex
        # self.feature_df['timestamp'] = pd.to_datetime(self.feature_df['timestamp'])
        # self.feature_df.set_index('timestamp', inplace=True)

        # Iterate over each row in feature_df
        for index, row in self.feature_df.iterrows():
            # Convert the row values to a numpy array
            values = np.array(row).reshape(1, -1)
            # Assign the numpy array to the corresponding datetimeindex in features_matrix_dict
            self.features_matrix_dict[index] = values

    def auto_mapping(self, fault_feat_mat):
        """
        Generate a feature vector for a specific scenario.

        Parameters:
            normal_vector (numpy.array): Feature vector for normal operation.
            scenario (dict): Dictionary specifying the scenario and constant values.
            constant_values (dict): Dictionary mapping parameter names to constant values.

        Returns:
            numpy.array: Feature vector for the specified scenario.
        """
        for key, value in self.features_matrix_dict.items():
            scenario_vector = []  # Initialize as an empty list
            for hypothesis, features in fault_feat_mat.items():
                # Copy the normal vector to avoid modifying the original
                temp_vector = np.copy(value)
                # print("Temp", temp_vector)
                # Replace parameter values with non-zero values from the features matrix
                for idx, feature in np.ndenumerate(features):
                    if feature != 0.0:
                        temp_vector[idx] = feature  # + np.random.normal(0, 1e-2)

                # Append the modified temp_vector to scenario_vector
                scenario_vector.append(temp_vector)

            # Concatenate scenario_vector along the row axis
            scenario_vector = np.concatenate(scenario_vector, axis=0)
            # print("Scenario", scenario_vector)

            # Append scenario_vector to self.features_matrix_dict[key]
            self.features_matrix_dict[key] = np.vstack([value, scenario_vector])

    def mapping(self, fault_feat_mat):
        """
        This function maps the hypotheses to the sensors

        Args:
            fault_feat_mat (dict): A dictionary where the keys represent hypotheses and the values represent their corresponding values.

        """
        fault = np.vstack([v for v in fault_feat_mat.values()])
        print(fault)
        for key, value in self.features_matrix_dict.items():
            self.features_matrix_dict[key] = np.vstack([value, fault])

    def input_feature_vector(self, hypothesis_name, feature_vector):
        self.temp_features_matrix_dict[hypothesis_name] = feature_vector

    # Process test hypotheses surveyed time range

    def read_data_between_dates(self, sensors, start_date, end_date, output_folder):
        # Initialize an empty dictionary to store data frames for each sensor
        data_frames = {sensor: pd.DataFrame() for sensor in sensors}
        # print("Output folder", output_folder)
        # Iterate over each date in the specified range
        start_datetime = datetime.strptime(start_date, "%d-%m-%Y %H:%M:%S")
        end_datetime = datetime.strptime(end_date, "%d-%m-%Y %H:%M:%S")
        dates = pd.date_range(start=start_datetime, end=end_datetime, freq="D")

        for date in dates:
            # Format the file name pattern with the date and sensors
            process_day = date.strftime("%Y%m%d")
            file_name = process_day + "_UT{}" + ".csv"
            data_folder = output_folder.format(process_day)
            # Iterate over each sensor and read its corresponding CSV file
            for sensor in sensors:
                file_path = os.path.join(data_folder, file_name.format(sensor))
                # print("File path:", file_path)
                if os.path.exists(file_path):
                    # print("Reading file:", file_path)
                    # Read the CSV file and append its data to the corresponding sensor DataFrame
                    df = pd.read_csv(file_path)
                    data_frames[sensor] = pd.concat(
                        [data_frames[sensor], df], ignore_index=True
                    )
                else:
                    print("File does not exist:", file_path)
        # print("Data Frames:", data_frames)
        return data_frames

    def feature_matrix_hypothesis(self, data, parameters, start_time, end_time):
        """
        This function creates a feature matrix for faulty sensors
        """
        time_header = "time_stamp"
        # print("Feature Matrix Hypothesis", start_time, end_time)
        # print("Start Time", start_time)
        # print("End Time", end_time)
        operation_data = {}
        for key, value in data.items():
            operation_data[key] = value[parameters][
                (value[time_header] > convert_to_unix(start_time))
                & (value[time_header] < convert_to_unix(end_time))
            ]
        # print(operation_data)
        # Filter out rows with zero elements
        for key, value in operation_data.items():
            operation_data[key] = value[(value != 0).all(axis=1)]

        if len(operation_data) == 1:
            key, value = next(iter(operation_data.items()))
            feat_mat_fault = value.mean(axis=0).to_numpy()
        else:
            mean_operation_data = {
                key: value.mean(axis=0) for key, value in operation_data.items()
            }
            # Calculate the mean of the means
            mean_values = [value.to_numpy() for value in mean_operation_data.values()]
            feat_mat_fault = np.mean(mean_values, axis=0)

        return feat_mat_fault

    def process_hypothesis(
        self, sensors, start_time, end_time, output_folder, parameters
    ):
        data = self.read_data_between_dates(
            sensors, start_time, end_time, output_folder
        )
        # print("Finished step 1")
        feat_mat_hypo = self.feature_matrix_hypothesis(
            data, parameters, start_time, end_time
        )
        return feat_mat_hypo

    def process_hypotheses(self, hypotheses_names, output_folder, parameters):
        for hypothesis in hypotheses_names:
            if hypothesis in list(self.hypotheses_info.keys()):
                # print("hypothesis", hypothesis)
                feat_mat_hypothesis = []
                # Get the averaged characteristic feature vector for each sampling
                for sampling in self.hypotheses_info[hypothesis]:
                    # print("sampling", sampling)
                    # for sensors, time_range in sampling:
                    sensors = sampling[0]
                    time_range = sampling[1]
                    start_time, end_time = time_range
                    feat_mat_hypothesis = self.process_hypothesis(
                        sensors, start_time, end_time, output_folder, parameters
                    )

                # Get the averaged characteristic feature vector for each hypothesis
                # Do something with feat_mat_fault
                self.input_feature_vector(hypothesis, feat_mat_hypothesis)

    # Final output of class

    def get_features_matrix_dict(self):
        return self.features_matrix_dict

    def get_features_matrix(self):
        """Get the concatenated features matrix

        Returns:
            ndarray: Returns the concatenated features matrix
        """
        if self.temp_features_matrix_dict:
            self.features_matrix_dict = mirror_and_populate_dict(
                self.hypotheses_info, self.temp_features_matrix_dict
            )
            print("Features Matrix Dict", self.features_matrix_dict)
            self.features_matrix_np = np.vstack(
                [v for v in self.features_matrix_dict.values() if v is not None]
            )
        return self.features_matrix_np
