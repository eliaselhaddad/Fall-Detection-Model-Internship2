from datetime import datetime
import os

import joblib
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from loguru import logger

from src.helper_functions.model_helper_functions import ModelHelpingFunctions


class FallPrediction:
    def __init__(self, model_path: str, path_to_scaler: str, path_to_data: str):
        self.model = load_model(model_path)
        self.scaler_path = path_to_scaler
        self.model_helper = ModelHelpingFunctions()
        self.data = pd.read_csv(path_to_data)
        self.minimun_data_size = 50
        self.probability_threshold = 0.5
        self.padding_size = 108
        self.configurations = [
            (50, 49),
            (40, 59),
            (30, 69),
            (20, 79),
            (60, 39),
            (70, 29),
            (80, 19),
        ]
        self.scaler = None
        # self.data = self.crop_data(self.df)

    def crop_data(self, data: pd.DataFrame):
        try:
            if "jerk" not in data.columns:
                self.model_helper.log_error("Data does not contain 'jerk' column")
                raise ValueError("Data does not contain 'jerk' column")

            max_jerk_index = data["jerk"].idxmax()
            self.model_helper.log_info(f"Max jerk index: {max_jerk_index}")

            cropped_datasets = []

            for left, right in self.configurations:
                logger.info(f"Left: {left}, Right: {right}")
                start_index = max(0, max_jerk_index - left)
                end_index = min(len(data), max_jerk_index + right)
                logger.info(f"Start index: {start_index}, End index: {end_index}")
                cropped_data = data.iloc[start_index:end_index]
                cropped_datasets.append(cropped_data)

            return cropped_datasets

        except Exception as e:
            self.model_helper.log_exception(f"Error cropping data: {e}")
            raise Exception(f"Error cropping data: {e}")

    def check_data_size(self, min_size):
        if len(self.data) < min_size:
            self.model_helper.log_error(
                f"Data length is less than {min_size} required for prediction"
            )
            raise ValueError(
                f"Data length is less than {min_size} required for prediction"
            )
        self.model_helper.log_info(f"Data length: {len(self.data)} passed the check")

    def load_scaler(self, scaler_path):
        try:
            self.model_helper.log_info(f"Loading scaler from {scaler_path}")
            with open(scaler_path, "rb") as file:
                self.scaler = joblib.load(file)
            return self.scaler
        except FileNotFoundError as e:
            self.model_helper.log_error(f"File not found: {e}")
            raise FileNotFoundError(f"File not found: {e}")
        except Exception as e:
            self.model_helper.log_error(f"Error loading scaler: {e}")
            raise Exception(f"Error loading scaler: {e}")

    def convert_to_numpy(self):
        try:
            self.model_helper.log_info("Converting to numpy")
            self.data = self.data.drop(columns=["fall_state"])
            self.model_helper.log_info(f"Data shape: {self.data.shape}")
            self.data = self.data.to_numpy()
            return self.data
        except Exception as e:
            self.model_helper.log_exception(f"Error converting to numpy: {e}")
            raise Exception(f"Error converting to numpy: {e}")

    def pad_data(self):
        try:
            self.model_helper.log_info("Padding data")
            data = self.convert_to_numpy()
            self.data = pad_sequences(
                [data],
                maxlen=self.padding_size,
                dtype="float32",
                padding="post",
                truncating="post",
            )
            return self.data
        except Exception as e:
            self.model_helper.log_exception(f"Error padding data: {e}")
            raise Exception(f"Error padding data: {e}")

    def scale_data(self):
        try:
            data = self.pad_data()
            scaler = self.load_scaler(self.scaler_path)

            self.model_helper.log_info("Scaling data")
            self.data = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(
                data.shape
            )
            self.model_helper.log_info(f"Scaled data: {self.data.shape}")
            return self.data
        except Exception as e:
            self.model_helper.log_exception(f"Error scaling data: {e}")
            raise Exception(f"Error scaling data: {e}")

    def predict(self):
        try:
            data = self.scale_data()
            return self.model.predict(data)
        except Exception as e:
            self.model_helper.log_exception(f"Error predicting: {e}")
            raise Exception(f"Error predicting: {e}")

    def predict_fall(self):
        self.check_data_size(self.minimun_data_size)

        try:
            prediction = self.predict()
            probability = prediction[0][0]

            if probability > self.probability_threshold:
                self.model_helper.log_warning(
                    f"Fall detected with probability: {probability}"
                )
                self.model_helper.log_warning("Fall detected")
            else:
                self.model_helper.log_success(
                    f"No fall detected with probability: {probability}"
                )
                self.model_helper.log_success("No fall detected")
        except Exception as e:
            self.model_helper.log_exception(f"Error predicting fall: {e}")
            raise Exception(f"Error predicting fall: {e}")


def get_latest_model_date(directory_path):
    try:
        logger.info(f"Scanning directory: {directory_path} for latest model")

        all_items = os.listdir(directory_path)
        dates = [
            datetime.strptime(item, "%Y-%m-%d")
            for item in all_items
            if os.path.isdir(os.path.join(directory_path, item))
        ]

        if not dates:
            logger.error(f"No model found in directory {directory_path}")
            raise FileNotFoundError("No model found in directory")

        latest_date = max(dates).strftime("%Y-%m-%d")
        logger.info(f"Latest model found: {latest_date}")
        return latest_date
    except Exception as e:
        logger.exception(f"Error getting latest model date: {e}")
        raise Exception(f"Error getting latest model date: {e}")


# def main():
#     date = get_latest_model_date("models/model/")
#     model_path = f"models/model/{date}/fall_detection_model.keras"
#     scaler_path = "models/scaler/scaler.pkl"
#     data_path = "data/sample_cleaned/merged_data.csv"

#     fp = FallPrediction(model_path, scaler_path, data_path)
#     cropped_datasets = fp.crop_data(fp.df)
#     # fp.crop_data(fp.data)
#     # fp.predict_fall()

#     for cropped_data in cropped_datasets:
#         fp.data = cropped_data  # Update the data attribute with each cropped dataset
#         fp.predict_fall()


def main():
    date = get_latest_model_date("models/model/")
    model_path = f"models/model/{date}/fall_detection_model.keras"
    scaler_path = "models/scaler2/scaler.pkl"
    data_path = "data/sample_cleaned/merged_data.csv"

    fp = FallPrediction(model_path, scaler_path, data_path)
    # fp.crop_data(fp.data)
    fp.predict_fall()


if __name__ == "__main__":
    main()
