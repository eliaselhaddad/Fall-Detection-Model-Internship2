import pandas as pd
from pathlib import Path
import logging
from typing import List
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)  # Log messages will include the time at which the message was logged, the log level, and the message itself.


class CSVFileManager:
    """Handles file operations, such as finding and saving CSV files."""

    def __init__(self, input_directory: Path, output_directory: Path):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def find_csv_files(self) -> list[Path]:
        logging.info(f"Searching for CSV files in directory: {self.input_directory}")
        csv_files = list(self.input_directory.glob("*.csv"))
        logging.info(f"Found {len(csv_files)} CSV files")
        return csv_files

    def save_csv(self, df, output_file_name: str) -> None:
        if df.empty:
            logging.info("No data to save.")
            return

        self.output_directory.mkdir(parents=True, exist_ok=True)
        output_path = self.output_directory / output_file_name
        df.to_csv(output_path, index=False)
        logging.info(f"CSV file saved to {output_path}")


class CSVPreprocessor:
    """Reads and preprocesses CSV files."""

    def process(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.drop(columns=["timestamp", "time_interval"], errors="ignore", inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df


class CSVMerger:
    """Merges multiple CSV files into one."""

    def merge(self, data_frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not data_frames:
            logging.info("No data frames to merge.")
            return pd.DataFrame()

        merged_df = pd.concat(data_frames, ignore_index=True)
        logging.info("CSV files merged successfully.")
        return merged_df


def main():
    input_dir = Path("data/processed/")
    output_dir = Path("data/cleaned/")
    output_name = "merged_data.csv"

    file_manager = CSVFileManager(input_dir, output_dir)
    preprocessor = CSVPreprocessor()
    merger = CSVMerger()

    csv_files = file_manager.find_csv_files()
    processed_files = [preprocessor.process(file) for file in csv_files]
    merged_df = merger.merge(processed_files)
    file_manager.save_csv(merged_df, output_name)


if __name__ == "__main__":
    main()
