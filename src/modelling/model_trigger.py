from datetime import datetime
import os
import numpy as np
import pandas as pd
from loguru import logger
import dataclasses
from src.tools.acceleration import Acceleration
from collections import deque
from src.modelling.predict_stream import PredictStream


class ModelTrigger:
    @staticmethod
    def get_latest_model_date(directory_path: str):
        try:
            all_items = os.listdir(directory_path)
            dates = [
                datetime.strptime(item, "%Y-%m-%d").date()
                for item in all_items
                if os.path.isdir(os.path.join(directory_path, item))
            ]

            if not dates:
                logger.error(f"No model found in directory {directory_path}")
                raise FileNotFoundError("No model found in directory")

            latest_date = max(dates)
            logger.info(f"Latest model date: {latest_date}")
            return latest_date.strftime("%Y-%m-%d")
        except Exception as e:
            logger.error(f"Error getting latest model date: {e}")
            raise

    def __init__(self, csv_row_limit: int = 104, csv_overlap: int = 13):
        self.csv_buffer_size = 5
        self.csv_row_limit = csv_row_limit
        self.csv_overlap = csv_overlap
        self.csv_buffer = deque(maxlen=self.csv_buffer_size)
        self.current_data = pd.DataFrame(
            columns=["timestamp", "timestamp_local", "ax", "ay", "az", "fall_state"]
        )
        self.g_force_threshold = 12
        self.csv_file_counter = 1
        self.latest_model_date = self.get_latest_model_date("models/model/")

    def update_data_window(self, acceleration: Acceleration):
        new_row = pd.DataFrame(
            [dataclasses.asdict(acceleration)], columns=self.current_data.columns
        )
        self.current_data = pd.concat([self.current_data, new_row], ignore_index=True)

        if len(self.current_data) >= self.csv_row_limit:
            self.csv_buffer.append(
                (f"file{self.csv_file_counter}", self.current_data.copy())
            )
            logger.info(f"Added file{self.csv_file_counter} to buffer")
            self.csv_file_counter += 1
            if len(self.csv_buffer) == self.csv_buffer_size and self.should_trigger(
                self.current_data
            ):
                logger.info(f"Triggered at {acceleration.timestamp}")
                self.run_prediction_on_triggered_files()
            self.current_data = self.current_data.tail(self.csv_overlap).reset_index(
                drop=True
            )

    def run_prediction_on_triggered_files(self):
        current_index = len(self.csv_buffer) - 1
        files_to_predict = [
            df
            for _, df in list(self.csv_buffer)[
                max(0, current_index - 4) : current_index + 1
            ]
        ]
        predictions = [self.predict_file(file_data) for file_data in files_to_predict]

        for i, prediction in enumerate(predictions):
            logger.info(f"Prediction for file{current_index - 4 + i}: {prediction}")

        self.print_prediction(predictions)
        # if "Fall" in predictions:
        #     final_prediction = "Fall"
        # else:
        #     final_prediction = "Not Fall"

        # logger.info(f"Aggregated prediction: {final_prediction}")

        # logger.info(f"Aggregated prediction: {final_prediction}")
        # for file_name, file_data in files_to_predict:
        #     prediction = self.predict_file(file_data)
        # logger.info(f"Prediction for {file_name}: {prediction}")

    def print_prediction(self, predictions):
        if "Fall" in predictions:
            final_prediction = "Fall"
        else:
            final_prediction = "Not Fall"
        logger.info(f"Aggregated prediction: {final_prediction}")

    def predict_file(self, df: pd.DataFrame):
        predict_stream = PredictStream(df, self.latest_model_date)
        return predict_stream.predict()

    def calculate_g_force(self, df: pd.DataFrame):
        try:
            g_force = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
            return g_force
        except KeyError:
            logger.error("Columns not found")
            raise

    def should_trigger(self, df: pd.DataFrame):
        g_force = self.calculate_g_force(df)
        return g_force.max() > self.g_force_threshold

    def check_and_print_removed_file(self):
        removed_file, removed_data = self.csv_buffer[0]
        if self.should_trigger(removed_data):
            status = "triggered"
        else:
            status = "not_triggered"
        logger.info(f"{removed_file} {status}")
