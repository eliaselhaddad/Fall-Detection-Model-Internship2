"""Run make test_all in the terminal to run all the tests"""

import unittest
from pathlib import Path
from typing import List
import pandas as pd
import shutil
import numpy as np
from unittest.mock import patch, MagicMock

from src.processing.merge_csv_files import CSVFilesMerger


class TestCSVFilesMerger(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir: Path = Path("temp_test_dir")
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        cls.input_dir: Path = cls.temp_dir / "input"
        cls.output_dir: Path = cls.temp_dir / "output"
        cls.input_dir.mkdir(exist_ok=True)
        cls.output_dir.mkdir(exist_ok=True)

        # Create sample CSV file
        sample_data: pd.DataFrame = pd.DataFrame(
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
        sample_data.to_csv(cls.input_dir / "sample1.csv", index=False)
        sample_data.to_csv(cls.input_dir / "sample2.csv", index=False)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir)

    @patch("src.processing.merge_csv_files.CSVFilesMerger.merge_and_process_files")
    def test_save_merged_csv(self, mock_merge: pd.DataFrame) -> None:
        # Setup mock
        mock_df = pd.DataFrame({"column1": [1, 2], "column2": [3, 4]})
        mock_merge.return_value = mock_df

        # Instantiate class and call the method under test
        merger = CSVFilesMerger(self.input_dir, self.output_dir, "merged.csv")
        merger.save_merged_csv(mock_df)

        # Verify the file was saved
        saved_file_path = self.output_dir / "merged.csv"
        self.assertTrue(saved_file_path.exists(), "Merged CSV file was not saved.")

        # Verify the content of the saved file matches mock_df
        saved_df = pd.read_csv(saved_file_path)
        pd.testing.assert_frame_equal(saved_df, mock_df, check_dtype=False)

        # Cleanup after test
        saved_file_path.unlink()

    def test_merged_csv_columns_count(self) -> None:
        merger: CSVFilesMerger = CSVFilesMerger(
            self.input_dir, self.output_dir, "merged.csv"
        )
        merged_df: pd.DataFrame = merger.merge_and_process_files()
        expected_columns_count = 22
        actual_columns_count = len(merged_df.columns)
        self.assertEqual(
            actual_columns_count,
            expected_columns_count,
            f"Expected {expected_columns_count} columns, but found {actual_columns_count}.",
        )


if __name__ == "__main__":
    unittest.main()
