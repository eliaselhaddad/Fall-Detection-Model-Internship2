import joblib
from loguru import logger
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from src.processing.data_validator import DataValidator
from src.processing.motion_features import MotionFeatureCalculator


class PredictStream:
    model = None
    scaler = None

    def __init__(self, data: pd.DataFrame, model_date: str):
        self.data = data.drop(columns=["g_force"], errors="ignore")
        self.data_validator = DataValidator()
        self.motion_feature_calculator = MotionFeatureCalculator()
        self.path_to_scaler = "models/scaler/scaler.pkl"
        self.model_path = f"models/model/{model_date}/fall_detection_model.keras"
        if PredictStream.scaler is None:
            PredictStream.scaler = self.load_scaler(self.path_to_scaler)
        if PredictStream.model is None:
            PredictStream.model = load_model(self.model_path)

    def process_data(self) -> pd.DataFrame:
        if "timestamp" not in self.data.columns:
            logger.error("Missing 'timestamp' column in data.")
            raise KeyError("Missing 'timestamp' column in data.")
        validated_data = self.data_validator.validate(
            data=self.data,
            required_accelerator_columns=self.data.columns,
        )
        processed_data = self.motion_feature_calculator.calculate_all_features(
            validated_data, "timestamp", ["ax", "ay", "az"]
        )
        processed_data = processed_data.drop(
            columns=["timestamp", "time_interval"], errors="ignore"
        )
        processed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        return processed_data.dropna()

    def prepare_data_for_prediction(self) -> np.ndarray:
        data = self.process_data()
        return data.drop(columns=["fall_state"]).to_numpy()

    @staticmethod
    def load_scaler(scaler_path):
        try:
            with open(scaler_path, "rb") as file:
                return joblib.load(file)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise

    def scale_data(self, data):
        data_shape = data.shape
        return PredictStream.scaler.transform(data.reshape(-1, data_shape[-1])).reshape(
            data_shape
        )

    @staticmethod
    def pad_data(data, maxlen=108):
        return pad_sequences(
            [data], maxlen=maxlen, dtype="float32", padding="post", truncating="post"
        )[0]

    def predict(self):
        data = self.prepare_data_for_prediction()
        padded_data = self.pad_data(data)
        scaled_data = self.scale_data(padded_data)
        probability = PredictStream.model.predict(np.array([scaled_data]))[0][0]

        return "Fall" if probability > 0.5 else "Not Fall"
