import json
import os

import joblib
import numpy as np
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from src.helper_functions.model_helper_functions import (
    ModelHelpingFunctions,
)


class ModelUtilities:
    def __init__(self):
        self.model_helper = ModelHelpingFunctions()

    def filter_and_load_data(self, file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(file_path)
            if len(data) <= 110:
                data = data.drop(columns=["fall_state"])
                data = data.dropna()
                return data.to_numpy()
            else:
                self.model_helper.log_info(
                    f"Skipping file {file_path} as it exceeds the length limit"
                )
                return None
        except Exception as e:
            self.model_helper.log_error(f"Error processing file {file_path}: {e}")
            raise e

    def load_data_to_numpy_arrays(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        self.model_helper.log_info("Loading data into numpy arrays")
        fall_sequences = []
        non_fall_sequences = []
        try:
            for file in os.listdir(path):
                self.model_helper.log_info(f"Loading file {file}")
                file_path = os.path.join(path, file)
                if file.startswith("fall"):
                    sequence = self.filter_and_load_data(file_path)
                    if sequence is not None:
                        fall_sequences.append(sequence)
                elif file.startswith("non_fall"):
                    sequence = self.filter_and_load_data(file_path)
                    if sequence is not None:
                        non_fall_sequences.append(sequence)
        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error loading data: {e}")

        self.model_helper.log_info("Data loaded into numpy arrays")
        return np.array(fall_sequences, dtype=object), np.array(
            non_fall_sequences, dtype=object
        )

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            self.model_helper.log_info("Loading data from data/seq/ for training")
            fall_data, non_fall_data = self.load_data_to_numpy_arrays("data/seq/")
            self.model_helper.log_info(
                f"Data loaded length fall: {len(fall_data)} non-fall: {len(non_fall_data)}  total: {len(fall_data) + len(non_fall_data)}"
            )
            return fall_data, non_fall_data
        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error loading data: {e}")

    def prepare_labels(self, fall_data, non_fall_data) -> tuple[list[int], list[int]]:
        try:
            self.model_helper.log_info("Preparing labels for each numpy array sequence")
            fall_labels = [1 for _ in fall_data]
            non_fall_labels = [0 for _ in non_fall_data]
            self.model_helper.log_info(
                f"Labels prepared: fall: {len(fall_labels)} non-fall: {len(non_fall_labels)}, total: {len(fall_labels) + len(non_fall_labels)}"
            )
            return fall_labels, non_fall_labels
        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error preparing labels: {e}")

    def pad_sequences(self, all_sequences: list) -> np.ndarray:
        try:
            self.model_helper.log_info("Padding sequences to the same length")
            return pad_sequences(
                all_sequences, dtype="float32", padding="post", truncating="post"
            )
        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error padding sequences: {e}")

    def scale_sequences(self, padded_sequences: np.ndarray) -> np.ndarray:
        try:
            self.model_helper.log_info("Scaling sequences using MinMaxScaler")
            scaler = MinMaxScaler()

            scaled_data = scaler.fit_transform(
                padded_sequences.reshape(-1, padded_sequences.shape[-1])
            ).reshape(padded_sequences.shape)

            scaler_directory = "models/scaler2"
            if not os.path.exists(scaler_directory):
                os.makedirs(scaler_directory)

            scaler_path = os.path.join(scaler_directory, "scaler.pkl")
            self.model_helper.log_info(f"Saving scaler to {scaler_path}")
            with open(scaler_path, "wb") as file:
                joblib.dump(scaler, file)
            self.model_helper.log_info("Scaler saved")
            return scaled_data
        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error scaling sequences: {e}")

    def prep_train_val_test_data(
        self, scaled_sequences: np.ndarray, all_labels: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], list[int]]:
        try:
            self.model_helper.log_info(
                "Splitting data into train, validation, and test sets"
            )
            (
                train_sequences,
                test_sequences,
                train_labels,
                test_labels,
            ) = train_test_split(
                scaled_sequences, all_labels, test_size=0.2, random_state=42
            )
            train_sequences, val_sequences, train_labels, val_labels = train_test_split(
                train_sequences, train_labels, test_size=0.2, random_state=42
            )
            self.model_helper.log_info("Data prepared")
            return (
                train_sequences,
                val_sequences,
                test_sequences,
                train_labels,
                val_labels,
                test_labels,
            )
        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error preparing train, validation, and test data: {e}")

    def get_model_hyperparameters(self, parameter_name: str) -> dict:
        self.model_helper.log_info(f"Getting hyperparameters from {parameter_name}")
        with open(f"{parameter_name}", "r") as file:
            data = json.load(file)
            return data

    def train_model(
        self,
        model: tf.keras.Model,
        train_sequences: np.ndarray,
        train_labels: list,
        val_sequences: np.ndarray,
        val_labels: list,
    ) -> tf.keras.callbacks.History:
        try:
            self.model_helper.log_info(
                "Training model with training sequences and labels"
            )
            return model.fit(
                train_sequences,
                np.array(train_labels),
                epochs=100,
                validation_data=(val_sequences, np.array(val_labels)),
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=7)],
                shuffle=True,
            )

        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error training model: {e}")

    def train_final_model(
        self, model: tf.keras.Model, train_sequences: np.ndarray, train_labels: list
    ) -> tf.keras.callbacks.History:
        try:
            self.model_helper.log_info(
                "Training model with training sequences and labels"
            )
            return model.fit(
                train_sequences,
                np.array(train_labels),
                epochs=32,
                shuffle=True,
            )

        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error training model: {e}")

    def evaluate_model(
        self, model: tf.keras.Model, test_sequences: np.ndarray, test_labels: list
    ) -> list[float]:
        try:
            self.model_helper.log_info("Evaluating model")
            return model.evaluate(test_sequences, np.array(test_labels))
        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error evaluating model: {e}")
