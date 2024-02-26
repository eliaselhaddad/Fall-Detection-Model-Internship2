"""Run make test_all in the terminal to run all the tests"""

import unittest
from pathlib import Path
import pandas as pd
import shutil
import numpy as np
from unittest.mock import patch
from src.processing.csv_processing_pipeline import (
    CSVFileManager,
    CSVPreprocessor,
    CSVMerger,
)


class TestCSVProcessingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir: Path = Path("temp_test_dir")
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        cls.input_dir: Path = cls.temp_dir / "input"
        cls.output_dir: Path = cls.temp_dir / "output"
        cls.input_dir.mkdir(exist_ok=True)
        cls.output_dir.mkdir(exist_ok=True)

        # Create sample CSV file
        cls.sample_data: pd.DataFrame = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2021-01-01", periods=5, freq="T"),
                "ax": np.random.rand(5),
                "ay": np.random.rand(5),
                "az": np.random.rand(5),
                "fall_state": [0, 1, 0, 1, 0],
                "time_interval": np.linspace(0.1, 0.5, 5),
                "vx": np.random.rand(5),
                "vy": np.random.rand(5),
                "vz": np.random.rand(5),
                "dx": np.random.rand(5),
                "dy": np.random.rand(5),
                "dz": np.random.rand(5),
                "angle_xy": np.random.rand(5) * 360 - 180,
                "angle_yz": np.random.rand(5) * 360 - 180,
                "angle_zx": np.random.rand(5) * 360 - 180,
                "g_force": np.random.rand(5),
                "jerk": np.random.rand(5),
                "orientation_xy": np.random.rand(5) * 360 - 180,
                "orientation_yz": np.random.rand(5) * 360 - 180,
                "orientation_zx": np.random.rand(5) * 360 - 180,
                "magnitude_acceleration": np.random.rand(5),
                "magnitude_velocity": np.random.rand(5),
                "magnitude_displacement": np.random.rand(5),
                "impact_detection": [0, 0, 1, 0, 1],
            }
        )
        cls.sample_data.to_csv(cls.input_dir / "sample1.csv", index=False)
        cls.sample_data.to_csv(cls.input_dir / "sample2.csv", index=False)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir)

    def test_find_csv_files(self) -> None:
        # Arrange
        file_manager = CSVFileManager(self.input_dir, self.output_dir)

        # Act
        found_files = file_manager.find_csv_files()

        # Assert
        self.assertEqual(len(found_files), 2, "Should find two CSV files.")

    def test_csv_preprocessor(self) -> None:
        # Arrange
        preprocessor = CSVPreprocessor()
        file_path = self.input_dir / "sample1.csv"

        # Act
        processed_df = preprocessor.process(file_path)

        # Assert
        expected_columns = set(self.sample_data.columns) - {
            "timestamp",
            "time_interval",
        }
        self.assertTrue(
            set(processed_df.columns).issubset(expected_columns),
            "Processed DataFrame should not contain 'timestamp' or 'time_interval' columns.",
        )
        self.assertFalse(
            processed_df.isnull().values.any(),
            "Processed DataFrame should not contain any null values.",
        )

    def test_csv_merger(self) -> None:
        # Arrange
        preprocessor = CSVPreprocessor()
        merger = CSVMerger()
        processed_files = [
            preprocessor.process(self.input_dir / f)
            for f in ["sample1.csv", "sample2.csv"]
        ]

        # Act
        merged_df = merger.merge(processed_files)

        # Assert
        expected_row_count = sum(df.shape[0] for df in processed_files)
        self.assertEqual(
            merged_df.shape[0],
            expected_row_count,
            "Merged DataFrame should have the combined row count of all processed files.",
        )


if __name__ == "__main__":
    unittest.main()
