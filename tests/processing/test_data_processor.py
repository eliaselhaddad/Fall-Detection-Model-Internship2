import os
import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import io
from src.processing.data_processor import DataProcessor
from pathlib import Path
import shutil

mock_data = """timestamp,timestamp_local,ax,ay,az,fall_state
1644144042.430037,2024-02-06 10:00:42.430037,-0.562333,8.858545,0.452260,0
1644144042.504867,2024-02-06 10:00:42.504867,-0.737016,8.805902,0.686765,0
1644144042.579836,2024-02-06 10:00:42.579836,-0.356543,9.480701,0.746587,0
1644144042.654867,2024-02-06 10:00:42.654867,-1.072023,9.566847,0.181861,0
1644144042.729828,2024-02-06 10:00:42.729828,0.485760,10.150716,1.799467,0"""


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure the 'processed' directory exists before any tests run
        cls.processed_dir = Path("processed")
        cls.processed_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Clean up by removing the 'processed' directory after all tests have run
        shutil.rmtree(cls.processed_dir)

    def setUp(self):
        # Setup for each test method
        self.processor = DataProcessor("raw", "processed")

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=mock_data)
    def test_load_data(self, mock_file, mock_exists):
        # Arrange
        filename = "test.csv"
        expected_data = pd.read_csv(io.StringIO(mock_data))
        # Act
        loaded_data = self.processor.load_data(filename)
        # Assert
        pd.testing.assert_frame_equal(loaded_data, expected_data)

    @patch("src.processing.data_processor.DataProcessor.load_data")
    def test_process_file(self, mock_load_data):
        # Arrange
        mock_load_data.return_value = pd.read_csv(io.StringIO(mock_data))
        # Act
        self.processor.process_file("test.csv")
        # Assert
        self.assertTrue((Path("processed") / "processed_test.csv").exists())

    def test_save_data(self):
        # Arrange
        data = pd.read_csv(io.StringIO(mock_data))
        # Act
        self.processor.save_data(data, "test.csv")
        # Assert
        self.assertTrue((Path("processed") / "test.csv").exists())


if __name__ == "__main__":
    unittest.main()
