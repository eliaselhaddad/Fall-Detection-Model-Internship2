import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from loguru import logger
import joblib


class ModelTraining:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.data = self.data[self.data["fall_state"].isin([0, 1])]
        self.counter = 0
        self.start_index = 0

    def split_csv(self):
        try:
            logger.info("Splitting CSV file into sequences")
            for i in range(1, len(self.data)):
                if (
                    self.data.iloc[i]["fall_state"]
                    != self.data.iloc[i - 1]["fall_state"]
                ):
                    logger.info(f"Splitting sequence at index {i}")
                    file_name = f"data/seq/{'non_fall' if self.data.iloc[i-1]['fall_state'] == 0 else 'fall'}_{self.counter}.csv"
                    self.data.iloc[self.start_index : i].to_csv(file_name, index=False)
                    self.counter += 1
                    self.start_index = i
        except Exception as e:
            logger.error(f"Error splitting CSV file: {e}")
            raise Exception(f"Error splitting CSV file: {e}")

        try:
            logger.info(f"Splitting sequence at index {len(self.data)}")
            file_name = f"data/seq/{'non_fall' if self.data.iloc[-1]['fall_state'] == 0 else 'fall'}_{self.counter}.csv"
            self.data.iloc[self.start_index :].to_csv(file_name, index=False)
            logger.info("CSV file split into sequences complete")
        except Exception as e:
            logger.error(f"Error splitting CSV file: {e}")
            raise Exception(f"Error splitting CSV file: {e}")

    def load_data_to_numpy_arrays(self, path):
        logger.info("Loading data into numpy arrays")
        fall_sequences = []
        non_fall_sequences = []
        try:
            for file in os.listdir(path):
                logger.info(f"Loading file {file}")
                file_path = os.path.join(path, file)
                if file.startswith("fall"):
                    data = pd.read_csv(file_path)
                    fall_sequences.append(data.to_numpy())
                elif file.startswith("non_fall"):
                    data = pd.read_csv(file_path)
                    non_fall_sequences.append(data.to_numpy())
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise Exception(f"Error loading data: {e}")

        logger.info("Data loaded into numpy arrays")
        return np.array(fall_sequences, dtype=object), np.array(
            non_fall_sequences, dtype=object
        )

    def load_data(self):
        try:
            logger.info("Loading data")
            fall_data, non_fall_data = self.load_data_to_numpy_arrays("data/seq/")
            return fall_data, non_fall_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise Exception(f"Error loading data: {e}")

    def prepare_labels(self, fall_data, non_fall_data):
        try:
            logger.info("Preparing labels")
            fall_labels = [1 for _ in fall_data]
            non_fall_labels = [0 for _ in non_fall_data]
            return fall_labels, non_fall_labels
        except Exception as e:
            logger.error(f"Error preparing labels: {e}")
            raise Exception(f"Error preparing labels: {e}")

    def pad_sequences(self, all_sequences):
        try:
            logger.info("Padding sequences")
            return pad_sequences(
                all_sequences, dtype="float32", padding="post", truncating="post"
            )
        except Exception as e:
            logger.error(f"Error padding sequences: {e}")
            raise Exception(f"Error padding sequences: {e}")

    def scale_sequences(self, padded_sequences):
        try:
            logger.info("Scaling sequences")
            scaler = MinMaxScaler()

            scaled_data = scaler.fit_transform(
                padded_sequences.reshape(-1, padded_sequences.shape[-1])
            ).reshape(padded_sequences.shape)

            scaler_directory = "models/scaler"
            if not os.path.exists(scaler_directory):
                os.makedirs(scaler_directory)

            scaler_path = os.path.join(scaler_directory, "scaler.pkl")
            logger.info(f"Saving scaler to {scaler_path}")
            with open(scaler_path, "wb") as file:
                joblib.dump(scaler, file)
            logger.info("Scaler saved")
            return scaled_data
        except Exception as e:
            logger.error(f"Error scaling sequences: {e}")
            raise Exception(f"Error scaling sequences: {e}")

    def prep_train_val_test_data(self, scaled_sequences, all_labels):
        try:
            logger.info("Preparing train, validation, and test data")
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
            logger.info("Data prepared")
            return (
                train_sequences,
                val_sequences,
                test_sequences,
                train_labels,
                val_labels,
                test_labels,
            )
        except Exception as e:
            logger.error(f"Error preparing train, validation, and test data: {e}")
            raise Exception(f"Error preparing train, validation, and test data: {e}")

    def create_model(self, input_shape):
        try:
            logger.info("Creating model")
            model = Sequential(
                [
                    Bidirectional(
                        LSTM(64, return_sequences=True, input_shape=input_shape)
                    ),
                    Bidirectional(LSTM(32)),
                    Dense(64, activation="relu"),
                    Dropout(0.5),
                    Dense(1, activation="sigmoid"),
                ]
            )

            model.compile(
                loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
            return model
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise Exception(f"Error creating model: {e}")

    def train_model(
        self, model, train_sequences, train_labels, val_sequences, val_labels
    ):
        try:
            logger.info("Training model")
            return model.fit(
                train_sequences,
                np.array(train_labels),
                epochs=10,
                validation_data=(val_sequences, np.array(val_labels)),
                callbacks=[EarlyStopping(patience=3)],
            )
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise Exception(f"Error training model: {e}")

    def evaluate_model(self, model, test_sequences, test_labels):
        try:
            logger.info("Evaluating model")
            return model.evaluate(test_sequences, np.array(test_labels))
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise Exception(f"Error evaluating model: {e}")

    def save_model(self, model):
        try:
            logger.info("Saving model")
            date = datetime.now().strftime("%Y-%m-%d")
            directory = f"models/model/{date}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            model.save(f"{directory}/fall_detection_model.keras")
            logger.info("Model saved")

        except FileNotFoundError as e:
            logger.error(f"Error saving model: {e}")
            raise FileNotFoundError(f"Error saving model: {e}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
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
