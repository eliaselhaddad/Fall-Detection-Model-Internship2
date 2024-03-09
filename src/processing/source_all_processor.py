import argparse
import os
from pathlib import Path
import dataclasses

import numpy as np
import pandas as pd
from loguru import logger

from src.processing.data_validator import DataValidator
from src.processing.motion_features import MotionFeatureCalculator
from src.models.acceleration import Acceleration

"""
Som jag förstår det så används DataProcessor och sedan också data_processor_2.py för att processa data från en mapp
Du bör döpa åtminstone (...)_2 till något mer beskrivande så man förstår varför den ska användas också och vad den gör.

1. Glöm inte typning på funktioner och variabler
2. Sätt gärna funktionsordningen i klassen i klassen de används, t exså körs process_file först, lägg den direkt under init
Sedan is_already_processed, osv -> ökar läsbarheten och gör det lättare att förstå vad som händer
3. Sätt lowercase på variabler
4. Se kommentarer i koden längre ned.
"""


class DataProcessor:
    def __init__(self, raw: str, processed: str):
        self.raw = Path(raw)
        self.processed = Path(processed)
        self.required_accelerometer_columns = [
            field.name for field in dataclasses.fields(Acceleration)
        ]
        self.data_validator = DataValidator()
        self.motion_feature_calculator = MotionFeatureCalculator()

    def process_file(self, file: str) -> None:
        if self.is_already_processed(file):
            logger.info(f"File {file} already processed. Skipping...")
            return

        try:
            data = self.load_data(file)
            processed_data = self.process_data(data, file)
            self.save_data(processed_data, f"processed_{file}")
            logger.info(f"File {file} processed successfully")
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
        except Exception as e:
            logger.error(f"Error processing file: {e}")

    def is_already_processed(self, file: str) -> bool:
        processed_file_name = f"processed_{file}"
        processed_file_path = self.processed / processed_file_name
        return processed_file_path.exists()

    def load_data(self, filename: str) -> pd.DataFrame:
        logger.info(f"Loading data from {filename}")
        file_path = self.raw / filename
        if file_path.exists():
            logger.info(f"File found at {file_path}")
            return pd.read_csv(file_path)
        else:
            logger.error(f"No file found at {file_path}")
            raise FileNotFoundError(f"No file found at {file_path}")

    def process_data(self, data: pd.DataFrame, file: str) -> pd.DataFrame:
        try:
            logger.info(f"Validating data from {file}")
            validated_data = self.data_validator.validate(
                data, self.required_accelerometer_columns
            )
            logger.info("Processing data")
            processed_data = self.motion_feature_calculator.calculate_all_features(
                validated_data, "timestamp", ["ax", "ay", "az"]
            )
            logger.info("Data processed")
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise e
        return processed_data

    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        try:
            logger.info(f"Saving data to {filename}")
            save_path = self.processed / filename
            data.to_csv(save_path, index=False)
            logger.info(f"Data saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise e


def setup_directories(folder):
    if folder == "sample":
        path_to_raw = "data/sample_raw"
        path_to_processed = "data/sample_processed"
    elif folder == "data":
        path_to_raw = "data/raw"
        path_to_processed = "data/processed"
    elif folder == "data2":
        path_to_raw = "data2/raw"
        path_to_processed = "data2/processed"
    else:
        raise ValueError("Invalid folder name. Choose 'sample', 'data', or 'data2'.")

    for path in [path_to_raw, path_to_processed]:
        if not os.path.exists(path):
            logger.error(f"{path} directory does not exist")
            raise FileNotFoundError(f"{path} directory does not exist")

    return path_to_raw, path_to_processed


def main(folder):
    logger.info("Starting data processing")
    path_to_raw, path_to_processed = setup_directories(folder=folder)
    data_processor = DataProcessor(path_to_raw, path_to_processed)

    logger.info("Processing files")
    for file in os.listdir(path_to_raw):
        data_processor.process_file(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data")
    parser.add_argument(
        "folder",
        choices=["sample", "data", "data2"],
        help="Specify the folder to process",
    )
    args = parser.parse_args()
    main(args.folder)
