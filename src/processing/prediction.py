import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
from datetime import datetime
from loguru import logger


class FallPrediction:
    def __init__(self, model_path, path_to_scaler, path_to_data):
        self.model = load_model(model_path)
        self.scaler_path = path_to_scaler
        self.data = pd.read_csv(path_to_data)

    def load_scaler(self, scaler_path):
        try:
            logger.info(f"Loading scaler from {scaler_path}")
            with open(scaler_path, "rb") as file:
                self.scaler = joblib.load(file)
            return self.scaler
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise FileNotFoundError(f"File not found: {e}")
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            raise Exception(f"Error loading scaler: {e}")

    def convert_to_numpy(self):
        try:
            logger.info("Converting to numpy")
            self.data = self.data.to_numpy()
            return self.data
        except Exception as e:
            logger.error(f"Error converting to numpy: {e}")
            raise Exception(f"Error converting to numpy: {e}")

    def pad_data(self):
        try:
            logger.info("Padding data")
            data = self.convert_to_numpy()
            self.data = pad_sequences(
                [data], maxlen=1339, dtype="float32", padding="post", truncating="post"
            )
            return self.data
        except Exception as e:
            logger.error(f"Error padding data: {e}")
            raise Exception(f"Error padding data: {e}")

    def scale_data(self):
        try:
            data = self.pad_data()
            scaler = self.load_scaler(self.scaler_path)

            logger.info("Scaling data")
            self.data = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(
                data.shape
            )
            logger.info("Data scaled")
            return self.data
        except Exception as e:
            logger.error(f"Error scaling data: {e}")
            raise Exception(f"Error scaling data: {e}")

    def predict(self):
        try:
            data = self.scale_data()
            return self.model.predict(data)
        except Exception as e:
            logger.error(f"Error predicting: {e}")
            raise Exception(f"Error predicting: {e}")

    def predict_fall(self):
        try:
            if self.predict() > 0.5:
                logger.warning("Fall detected")
                print("Fall detected")
            else:
                logger.success("No fall detected")
                print("No fall detected")
        except Exception as e:
            logger.error(f"Error predicting fall: {e}")
            raise Exception(f"Error predicting fall: {e}")


def main():
    date_today = datetime.now().strftime("%Y-%m-%d")
    model_path = f"models/model/{date_today}/fall_detection_model.keras"
    scaler_path = "models/scaler/scaler.pkl"
    data_path = "data/sample_cleaned/merged_data.csv"
    fp = FallPrediction(model_path, scaler_path, data_path)
    fp.predict_fall()


if __name__ == "__main__":
    main()
