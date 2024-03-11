import argparse
import pandas as pd
from pathlib import Path
import logging
from typing import List
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)  # Log messages will include the time at which the message was logged, the log level, and the message itself.

"""
Sätt ordningen på funktionerna i klassen i den ordning de används, så blir det lättare att läsa
"""


class CSVFilesMerger:
    """
    The class is designed to:
    - Find all CSV files in a specified input directory.
    - Read and preprocess each file by removing specified columns ('timestamp' and 'time_interval')
      and handling 'inf' values and missing data.
    - Merge all preprocessed files into a single DataFrame.
    - Check for and log any missing values in the merged DataFrame.
    - Save the merged DataFrame to a new CSV file in a specified output directory.
    """

    def __init__(
        self, input_directory: Path, output_directory: Path, output_file_name: str
    ):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.output_file_name = output_file_name

    def find_csv_files(self) -> List[Path]:
        logging.info(f"Searching for CSV files in directory: {self.input_directory}")
        csv_files = list(self.input_directory.glob("*.csv"))
        logging.info(f"Found {len(csv_files)} CSV files")
        return csv_files

    def read_and_process_csv(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.drop(
            columns=["timestamp", "time_interval"], errors="ignore", inplace=True
        )  # Remove specified columns
        df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
        df.dropna(inplace=True)  # Remove rows with missing values
        return df

    def merge_and_process_files(self) -> pd.DataFrame:
        csv_files = self.find_csv_files()
        if not csv_files:
            logging.info("No CSV files found.")
            return pd.DataFrame()  # Return an empty DataFrame if no CSV files are found

        data_frames = [self.read_and_process_csv(file) for file in csv_files]
        merged_df = pd.concat(data_frames, ignore_index=True)
        logging.info("CSV files merged successfully.")
        return merged_df

    def check_for_missing_values(self, df: pd.DataFrame) -> None:
        if df.isnull().values.any():
            logging.warning("Missing values found in the merged CSV file.")
        else:
            logging.info("No missing values found in the merged CSV file.")

    def save_merged_csv(self, df: pd.DataFrame) -> None:
        if df.empty:
            logging.info("No data to save.")
            return

        self.output_directory.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the output directory exists
        output_path = self.output_directory / self.output_file_name
        df.to_csv(output_path, index=False)
        logging.info(f"Merged CSV file saved to {output_path}")


def main(use_sample=False):
    if use_sample:
        input_dir = Path("data/sample_processed/")
        output_dir = Path("data/sample_cleaned/")
    else:
        input_dir = Path("data/processed/")
        output_dir = Path("data/cleaned/")
    output_name = "merged_data.csv"

    merger = CSVFilesMerger(input_dir, output_dir, output_name)
    merged_df = merger.merge_and_process_files()
    merger.check_for_missing_values(merged_df)
    merger.save_merged_csv(merged_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CSV files")
    parser.add_argument(
        "--use_sample", action="store_true", help="Use sample data for final processing"
    )
    args = parser.parse_args()
    main(args.use_sample)
