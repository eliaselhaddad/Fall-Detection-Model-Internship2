import pandas as pd
from pathlib import Path

from src.helper_functions.model_helper_functions import ModelHelpingFunctions

"""
1. Denna bör döpas till något mer beskrivande :D 
2. Även här, sätt ordningen på funktionerna i klassen i den ordning de används, så blir det lättare att läsa
3. Se kommentarer i kod
"""
class DataProcessor2:
    def __init__(self, input_directory: str, output_directory: str):
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.fall_counter = 1000
        self.non_fall_counter = 1000
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.model_helper = ModelHelpingFunctions()

    # Enligt typningen returnerar denna inget, men vi returnerar en sträng
    def get_modified_filename(self, file_name: str) -> None:
        try:
            if file_name.startswith("processed_fall"):
                modified_file_name = f"fall_{self.fall_counter}.csv"
                self.model_helper.log_info(f"Saving file {modified_file_name}")
                self.fall_counter += 1
            elif file_name.startswith("processed_non_fall"):
                modified_file_name = f"non_fall_{self.non_fall_counter}.csv"
                self.model_helper.log_info(f"Saving file {modified_file_name}")
                self.non_fall_counter += 1
            else:
                modified_file_name = file_name
                self.model_helper.log_info(f"Saving file {modified_file_name}")
            return modified_file_name
        except Exception as e:
            self.model_helper.log_error(f"Error modifying file name: {e}")
            raise e

    def drop_timestamp_column(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.drop(columns=["timestamp", "time_interval"], errors="ignore")
            self.model_helper.log_info("Dropped timestamp and time_interval columns")
            return df
        except Exception as e:
            self.model_helper.log_error(f"Error dropping timestamp column: {e}")
            raise e

    def save_csv_into_folder(self, df: pd.DataFrame, file_name: str) -> None:
        try:
            modified_file_name = self.get_modified_filename(file_name)
            save_path = self.output_directory / modified_file_name
            self.model_helper.log_info(f"Saving file {save_path}")
            df.to_csv(save_path, index=False)
            self.model_helper.log_info(f"Data saved to {save_path}")
        except Exception as e:
            self.model_helper.log_exception(f"Error saving file: {e}")
            raise e

    # Ta bort s:et i slutet av funktionsnamnet
    def process_each_csv_files(self) -> None:
        for file_path in self.input_directory.glob("*.csv"):
            try:
                self.model_helper.log_info(f"Processing {file_path.name}")
                df = pd.read_csv(file_path)
                df = self.drop_timestamp_column(df)
                self.save_csv_into_folder(df, file_path.name)
            except FileNotFoundError as e:
                self.model_helper.log_error(f"File not found: {e}")
                raise FileNotFoundError(f"File not found: {e}")
            except Exception as e:
                self.model_helper.log_exception(f"Error processing file: {e}")
                raise e


def main():
    input_directory = "data2/processed/"
    output_directory = "data/seq/"

    data_processor = DataProcessor2(
        input_directory=input_directory, output_directory=output_directory
    )
    data_processor.process_each_csv_files()


if __name__ == "__main__":
    main()
