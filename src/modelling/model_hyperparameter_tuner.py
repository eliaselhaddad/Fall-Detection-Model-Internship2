import json
import os
from keras_tuner import HyperModel
from keras_tuner import HyperParameters
from keras_tuner import Objective
from keras_tuner import BayesianOptimization
import numpy as np
import tensorflow as tf


from src.helper_functions.model_helper_functions import (
    ModelHelpingFunctions,
)
from src.modelling.model_utilities import ModelUtilities


class ModelHyperparameterTuner(HyperModel):
    def __init__(self):
        self.epochs = 100
        self.seed = 42
        self.project_name = "fall_detection"
        self.directory = "tuner_dir"
        self.objective = "val_accuracy"
        self.patience = 7
        self.max_trials = 30
        self.model_helper = ModelHelpingFunctions()
        self.model_utilities = ModelUtilities()
        self.input_shape = None

    def _load_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], list[int]]:
        try:
            self.model_helper.log_info("Loading data for hyperparameter tuning...")
            fall_data, non_fall_data = self.model_utilities.load_data()
            fall_labels, non_fall_labels = self.model_utilities.prepare_labels(
                fall_data, non_fall_data
            )
            all_sequences = list(fall_data) + list(non_fall_data)
            all_labels = fall_labels + non_fall_labels
            padded_sequences = self.model_utilities.pad_sequences(all_sequences)

            scaled_sequences = self.model_utilities.scale_sequences(padded_sequences)
            (
                train_sequences,
                val_sequences,
                test_sequences,
                train_labels,
                val_labels,
                test_labels,
            ) = self.model_utilities.prep_train_val_test_data(
                scaled_sequences, all_labels
            )
            self.input_shape = train_sequences[0].shape
            return (
                train_sequences,
                val_sequences,
                test_sequences,
                train_labels,
                val_labels,
                test_labels,
            )
        except Exception as e:
            self.model_helper.log_exception(
                f"An error occurred while loading data for hyperparameter tuning: {e}"
            )

    def build(self, hp: HyperParameters) -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=self.input_shape))
        model.add(
            tf.keras.layers.Conv1D(
                filters=hp.Int("conv1_filters", min_value=64, max_value=512, step=32),
                kernel_size=hp.Int("conv1_kernel", min_value=2, max_value=5, step=1),
                activation="relu",
            )
        )
        model.add(
            tf.keras.layers.MaxPooling1D(
                pool_size=hp.Int("pool1_size", min_value=2, max_value=4, step=1)
            )
        ),
        model.add(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    hp.Int("lstm1_units", min_value=64, max_value=512, step=32),
                    return_sequences=True,
                )
            )
        )
        model.add(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    hp.Int("lstm2_units", min_value=64, max_value=512, step=32),
                    return_sequences=True,
                )
            )
        ),
        model.add(tf.keras.layers.GlobalAveragePooling1D()),

        model.add(
            tf.keras.layers.Dense(
                hp.Int("dense_units", min_value=64, max_value=512, step=32),
                activation="relu",
            )
        )

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        )
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
        return model

    def _model_tuner(
        self,
        objective: Objective,
        max_trials: int,
        seed: int,
        project_name: str,
        directory: str,
    ) -> BayesianOptimization:
        try:
            self.model_helper.log_info("Setting up Hyperparameter Tuner...")
            tuner = BayesianOptimization(
                hypermodel=self,
                objective=objective,
                max_trials=max_trials,
                seed=seed,
                project_name=project_name,
                directory=directory,
            )
            self.model_helper.log_info("Hyperparameter Tuner set up")
            return tuner
        except Exception as e:
            self.model_helper.log_exception(
                f"An error occurred while setting up Hyperparameter Tuner: {e}"
            )
            raise Exception(f"Error setting up Hyperparameter Tuner: {e}")

    def _best_params_search(
        self,
        tuner: BayesianOptimization,
        train_data: np.ndarray,
        train_labels: list,
        validation_data: np.ndarray,
        validation_labels: list,
        epochs: int,
    ) -> dict:
        try:
            self.model_helper.log_info("Searching for best parameters...")
            tuner.search(
                train_data,
                np.array(train_labels),
                epochs=epochs,
                validation_data=(validation_data, np.array(validation_labels)),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        self.objective, patience=self.patience
                    )
                ],
                verbose=1,
            )
            self.model_helper.log_info("Best parameters found")
            return tuner.get_best_hyperparameters()[0].values
        except Exception as e:
            self.model_helper.log_exception(
                f"An error occurred while searching for best parameters: {e}"
            )
            raise Exception(f"Error searching for best parameters: {e}")

    def _save_best_params(self, best_params: dict) -> None:
        try:
            self.model_helper.log_info("Saving best parameters...")
            path_to_save = "hyperparameters/best_params.json"
            os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

            with open(path_to_save, "w") as file:
                json.dump(best_params, file)
            self.model_helper.log_info("Best parameters saved")
        except Exception as e:
            self.model_helper.log_exception(
                f"An error occurred while saving best parameters: {e}"
            )
            raise Exception(f"Error saving best parameters: {e}")

    def orchestrate_tuning(self) -> dict:
        try:
            self.model_helper.log_info("Orchestrating hyperparameter tuning...")
            (
                train_sequences,
                val_sequences,
                test_sequences,
                train_labels,
                val_labels,
                test_labels,
            ) = self._load_data()
            self.model_helper.log_info(
                f"Train Sequences Shape: {train_sequences[0].shape}"
            )
            model = self.build(hp=HyperParameters())
            tuner = self._model_tuner(
                Objective(self.objective, direction="max"),
                max_trials=self.max_trials,
                seed=self.seed,
                project_name=self.project_name,
                directory=self.directory,
            )
            best_params = self._best_params_search(
                tuner=tuner,
                train_data=train_sequences,
                train_labels=train_labels,
                validation_data=val_sequences,
                validation_labels=val_labels,
                epochs=self.epochs,
            )
            self._save_best_params(best_params)
            self.model_helper.log_info("Hyperparameter tuning complete")
            return best_params
        except Exception as e:
            self.model_helper.log_exception(
                f"An error occurred while orchestrating hyperparameter tuning: {e}"
            )
            raise Exception(f"Error orchestrating hyperparameter tuning: {e}")


def main():
    model_tuner = ModelHyperparameterTuner()
    best_params = model_tuner.orchestrate_tuning()
    print(best_params)


if __name__ == "__main__":
    main()
