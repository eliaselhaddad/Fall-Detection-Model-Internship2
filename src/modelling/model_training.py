import os
from datetime import datetime

import numpy as np

import tensorflow as tf
from src.helper_functions.model_helper_functions import (
    ModelHelpingFunctions,
)
from src.modelling.model_utilities import ModelUtilities


class ModelTraining:
    def __init__(self):
        self.model_helper = ModelHelpingFunctions()
        self.model_utilities = ModelUtilities()

    def _create_model(self, input_shape: tuple[int]) -> tf.keras.Model:
        self.model_helper.log_info(f"Creating model with input shape: {input_shape}")
        self.model_helper.log_info("Getting hyperparameters for model creation")

        best_hyperparameters = self.model_utilities.get_model_hyperparameters(
            "hyperparameters/best_params.json"
        )
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Masking(mask_value=0.0, input_shape=input_shape),
                tf.keras.layers.Conv1D(
                    filters=best_hyperparameters["conv1_filters"],
                    kernel_size=best_hyperparameters["conv1_kernel"],
                    activation="relu",
                ),
                tf.keras.layers.MaxPooling1D(
                    pool_size=best_hyperparameters["pool1_size"]
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        best_hyperparameters["lstm1_units"], return_sequences=True
                    )
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        best_hyperparameters["lstm2_units"], return_sequences=True
                    )
                ),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(
                    best_hyperparameters["dense_units"], activation="relu"
                ),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=best_hyperparameters["learning_rate"]
        )
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
        self.model_helper.log_info("Model created")
        return model

    def _save_model(self, model: tf.keras.Model) -> None:
        try:
            self.model_helper.log_info("Saving model")
            date = datetime.now().strftime("%Y-%m-%d")
            directory = f"models/model/{date}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_save_path = f"{directory}/fall_detection_model.keras"
            model.save(model_save_path)
            self.model_helper.log_info("Model saved")

        except FileNotFoundError as e:
            self.model_helper.log_exception(e)
            raise FileNotFoundError(f"Error saving model: {e}")

        except Exception as e:
            self.model_helper.log_exception(e)
            raise Exception(f"Error saving model: {e}")

    def start(self) -> None:
        fall_data, non_fall_data = self.model_utilities.load_data()
        fall_labels, non_fall_labels = self.model_utilities.prepare_labels(
            fall_data, non_fall_data
        )
        all_sequences = list(fall_data) + list(non_fall_data)
        all_labels = fall_labels + non_fall_labels
        padded_sequences = self.model_utilities.pad_sequences(all_sequences)
        scaled_sequences = self.model_utilities.scale_sequences(padded_sequences)

        """
        # For training and evaluating the model, we can use the following code:
        (
            train_sequences,
            val_sequences,
            test_sequences,
            train_labels,
            val_labels,
            test_labels,
        ) = self.model_utilities.prep_train_val_test_data(scaled_sequences, all_labels)
        model = self._create_model(train_sequences[0].shape)
        history = self.model_utilities.train_model(
         model, train_sequences, train_labels, val_sequences, val_labels
        )

        self.model_utilities.evaluate_model(model, test_sequences, test_labels)
        """

        model = self._create_model(scaled_sequences[0].shape)
        history = self.model_utilities.train_final_model(
            model, scaled_sequences, np.array(all_labels)
        )
        self._save_model(model)


def main():
    model_training = ModelTraining()
    model_training.start()


if __name__ == "__main__":
    main()
