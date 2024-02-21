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


class DataProcessor:
    def __init__(self, raw, processed):
        self.raw = Path(raw)
        self.processed = Path(processed)
        self.required_accelerometer_columns = [
            field.name for field in dataclasses.fields(Acceleration)
        ]

    def load_data(self, filename):
        logger.info(f"Loading data from {filename}")
        file_path = self.raw / filename
        if file_path.exists():
            logger.info(f"File found at {file_path}")
            return pd.read_csv(file_path)
        else:
            logger.error(f"No file found at {file_path}")
            raise FileNotFoundError(f"No file found at {file_path}")

    def process_data(self, data, filename):
        try:
            validated_data = DataValidator(
                data, required_accelerator_columns=self.required_accelerometer_columns
            ).validate()
            motion_feature_calculator = MotionFeatureCalculator(
                validated_data, "timestamp", ["ax", "ay", "az"]
            )

            logger.info("Processing data")
            processed_data = motion_feature_calculator.calculate_all_features()
            logger.info("Data processed")
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise e
        return processed_data

    def is_already_processed(self, file):
        processed_file_name = f"processed_{file}"
        processed_file_path = self.processed / processed_file_name
        return processed_file_path.exists()

    def process_file(self, file):
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

    def save_data(self, data, filename):
        try:
            logger.info(f"Saving data to {filename}")
            save_path = self.processed / filename
            data.to_csv(save_path, index=False)
            logger.info(f"Data saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise e


def setup_directories(use_sample=False):
    PATH_TO_PROCESSED = "data/processed"
    PATH_TO_RAW = "data/raw"

    if use_sample:
        PATH_TO_RAW = "data/sample_raw"
        PATH_TO_PROCESSED = "data/sample_processed"
    else:
        PATH_TO_RAW = "data/raw"
        PATH_TO_PROCESSED = "data/processed"

    assert os.path.exists("data"), logger.error("Data directory does not exist")
    assert os.path.exists("data/processed"), logger.error(
        "Processed data directory does not exist"
    )
    assert os.path.exists("data/raw"), logger.error("Raw data directory does not exist")
    assert os.path.exists("data/sample_raw"), logger.error(
        "Sample raw data directory does not exist"
    )
    assert os.path.exists("data/sample_processed"), logger.error(
        "Sample processed data directory does not exist"
    )
    if not os.path.exists(PATH_TO_PROCESSED):
        logger.info(f"Creating processed data directory at {PATH_TO_PROCESSED}")
        os.makedirs(PATH_TO_PROCESSED)

    return PATH_TO_RAW, PATH_TO_PROCESSED


def main(use_sample):
    logger.info("Starting data processing")

    PATH_TO_RAW, PATH_TO_PROCESSED = setup_directories(use_sample=use_sample)
    data_processor = DataProcessor(PATH_TO_RAW, PATH_TO_PROCESSED)

    logger.info("Processing files")
    for file in os.listdir(PATH_TO_RAW):
        data_processor.process_file(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data")
    parser.add_argument(
        "--use_sample",
        action="store_true",
        help="Use sample data instead of full dataset",
    )
    args = parser.parse_args()
    main(args.use_sample)
