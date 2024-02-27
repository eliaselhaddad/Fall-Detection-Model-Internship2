import json
import os
from datetime import datetime

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

START_SEQUENCE_INDEX = 40
END_SEQUENCE_INDEX = 40


class ModelTraining:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.data = self.data[self.data["fall_state"].isin([0, 1])]
        self.counter = 0
        self.start_index = 0
        self.model_helper = ModelHelpingFunctions()

    def process_sequence(self, current_state, previous_state, i):
        try:
            self.model_helper.log_info(f"Processing sequence at index {i}")
            if current_state != previous_state:
                if previous_state == 1:
                    self.save_sequence(i, True)
                    self.in_fall_sequence = False
                else:
                    self.save_sequence(i, False)
                self.counter += 1
                self.start_index = i

            if current_state == 1 and not self.in_fall_sequence:
                self.in_fall_sequence = True
        except Exception as e:
            self.model_helper.log_error(e)

    def save_sequence(self, end_index, is_fall):
        try:
            self.model_helper.log_info("Saving sequence")
            file_type = "fall" if is_fall else "non_fall"
            file_name = f"data/seq/{file_type}_{self.counter}.csv"
            start_index = (
                max(self.start_index - START_SEQUENCE_INDEX, 0)
                if is_fall
                else self.start_index
            )
            end_index = (
                min(end_index + END_SEQUENCE_INDEX, len(self.data))
                if is_fall
                else end_index
            )
            self.data.iloc[start_index:end_index].to_csv(file_name, index=False)
        except Exception as e:
            self.model_helper.log_exception(e)

    def handle_last_sequence(self):
        try:
            self.model_helper.log_info(f"Splitting sequence at index {len(self.data)}")
            last_state = self.data.iloc[-1]["fall_state"]
            self.save_sequence(len(self.data), last_state == 1)
            self.model_helper.log_info("CSV file split into sequences complete")
        except Exception as e:
            self.model_helper.log_exception(e)

    def split_csv(self):
        try:
            self.model_helper.log_info(
                "Splitting CSV files into sequences of falls and non-falls"
            )
            self.in_fall_sequence = False
            for i in range(1, len(self.data)):
                current_state = self.data.iloc[i]["fall_state"]
                previous_state = self.data.iloc[i - 1]["fall_state"]
                self.process_sequence(current_state, previous_state, i)
        except Exception as e:
            self.model_helper.log_exception(e)

        try:
            self.handle_last_sequence()
        except Exception as e:
            self.model_helper.log_exception(e)

    def load_data_to_numpy_arrays(self, path):
        self.model_helper.log_info("Loading data into numpy arrays")
        fall_sequences = []
        non_fall_sequences = []
        try:
            for file in os.listdir(path):
                self.model_helper.log_info(f"Loading file {file}")
                file_path = os.path.join(path, file)
                if file.startswith("fall"):
                    data = pd.read_csv(file_path)
                    data = data.drop(columns=["fall_state"])
                    fall_sequences.append(data.to_numpy())
                elif file.startswith("non_fall"):
                    data = pd.read_csv(file_path)
                    data = data.drop(columns=["fall_state"])
                    non_fall_sequences.append(data.to_numpy())
        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error loading data: {e}")

        self.model_helper.log_info("Data loaded into numpy arrays")
        return np.array(fall_sequences, dtype=object), np.array(
            non_fall_sequences, dtype=object
        )

    def load_data(self):
        try:
            self.model_helper.log_info("Loading data from data/seq/ for training")
            fall_data, non_fall_data = self.load_data_to_numpy_arrays("data/seq/")
            return fall_data, non_fall_data
        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error loading data: {e}")

    def prepare_labels(self, fall_data, non_fall_data):
        try:
            self.model_helper.log_info("Preparing labels for each numpy array sequence")
            fall_labels = [1 for _ in fall_data]
            non_fall_labels = [0 for _ in non_fall_data]
            return fall_labels, non_fall_labels
        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error preparing labels: {e}")

    def pad_sequences(self, all_sequences):
        try:
            self.model_helper.log_info("Padding sequences to the same length")
            return pad_sequences(
                all_sequences, dtype="float32", padding="post", truncating="post"
            )
        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error padding sequences: {e}")

    def scale_sequences(self, padded_sequences):
        try:
            self.model_helper.log_info("Scaling sequences using MinMaxScaler")
            scaler = MinMaxScaler()

            scaled_data = scaler.fit_transform(
                padded_sequences.reshape(-1, padded_sequences.shape[-1])
            ).reshape(padded_sequences.shape)

            scaler_directory = "models/scaler"
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

    def prep_train_val_test_data(self, scaled_sequences, all_labels):
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

    def get_model_hyperparameters(self, parameter_name):
        self.model_helper.log_info(f"Getting hyperparameters for {parameter_name}")
        with open(f"models/hyperparameters/{parameter_name}", "r") as file:
            data = json.load(file)
            return data

    def create_model(self, input_shape):
        self.model_helper.log_info(f"Creating model with input shape: {input_shape}")
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=128,
                    kernel_size=5,
                    activation="relu",
                    input_shape=input_shape,
                ),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Conv1D(filters=320, kernel_size=2, activation="relu"),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Conv1D(filters=320, kernel_size=4, activation="relu"),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(288, return_sequences=True)
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(128, return_sequences=True)
                ),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(384, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.model_helper.log_info("Model created")
        return model

    def train_model(
        self, model, train_sequences, train_labels, val_sequences, val_labels
    ):
        try:
            self.model_helper.log_info(
                "Training model with training sequences and labels"
            )
            return model.fit(
                train_sequences,
                np.array(train_labels),
                epochs=100,
                validation_data=(val_sequences, np.array(val_labels)),
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
            )

        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error training model: {e}")

    def evaluate_model(self, model, test_sequences, test_labels):
        try:
            self.model_helper.log_info("Evaluating model")
            return model.evaluate(test_sequences, np.array(test_labels))
        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error evaluating model: {e}")

    def save_model(self, model):
        try:
            self.model_helper.log_info("Saving model")
            date = datetime.now().strftime("%Y-%m-%d")
            directory = f"models/model/{date}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            model.save(f"{directory}/fall_detection_model.keras")
            self.model_helper.log_info("Model saved")

        except FileNotFoundError as e:
            self.model_helper.log_exception(e)
            raise FileNotFoundError(f"Error saving model: {e}")

        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error saving model: {e}")


def main():
    model_training = ModelTraining("data/cleaned/merged_data.csv")
    model_training.split_csv()
    fall_data, non_fall_data = model_training.load_data()
    fall_labels, non_fall_labels = model_training.prepare_labels(
        fall_data, non_fall_data
    )
    all_sequences = list(fall_data) + list(non_fall_data)
    all_labels = fall_labels + non_fall_labels
    padded_sequences = model_training.pad_sequences(all_sequences)

    scaled_sequences = model_training.scale_sequences(padded_sequences)
    (
        train_sequences,
        val_sequences,
        test_sequences,
        train_labels,
        val_labels,
        test_labels,
    ) = model_training.prep_train_val_test_data(scaled_sequences, all_labels)
    model = model_training.create_model(train_sequences[0].shape)
    history = model_training.train_model(
        model, train_sequences, train_labels, val_sequences, val_labels
    )
    model_training.evaluate_model(model, test_sequences, test_labels)
    model_training.save_model(model)


if __name__ == "__main__":
    main()
