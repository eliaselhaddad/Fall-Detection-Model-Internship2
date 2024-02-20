import unittest
import pandas
from src.processing.data_validator import DataValidator


class TestDataValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = {
            "timestamp": [
                1644144042.430037,
                1644144042.504867,
                1644144042.579836,
                1644144042.654867,
                1644144042.729828,
            ],
            "timestamp_local": [
                "2024-02-06 10:00:42.430037",
                "2024-02-06 10:00:42.504867",
                "2024-02-06 10:00:42.579836",
                "2024-02-06 10:00:42.654867",
                "2024-02-06 10:00:42.729828",
            ],
            "ax": [-0.562333, -0.737016, -0.356543, -1.072023, 0.485760],
            "ay": [8.858545, 8.805902, 9.480701, 9.566847, 10.150716],
            "az": [0.452260, 0.686765, 0.746587, 0.181861, 1.799467],
            "fall_state": [0, 0, 0, 0, 0],
        }
        cls.df = pandas.DataFrame(data)
        cls.df["timestamp_local"] = pandas.to_datetime(cls.df["timestamp_local"])

        cls.validator = DataValidator(cls.df, "test.csv")

    def test_check_all_columns(self):
        expected = ["timestamp", "timestamp_local", "ax", "ay", "az", "fall_state"]
        self.assertEqual(self.validator.required_columns, expected)

    def test_check_null_values(self):
        self.assertIsNone(self.validator.check_null_values())

    def test_check_duplicated_rows(self):
        self.assertIsNone(self.validator.check_duplicated_rows())

    def test_drop_timestamp(self):
        self.assertIsNone(self.validator.drop_timestamp())
        self.assertNotIn("timestamp", self.validator.data.columns)

    def test_rename_timestamp_local(self):
        self.assertIsNone(self.validator.rename_timestamp_local())
        self.assertIn("timestamp", self.validator.data.columns)
        self.assertNotIn("timestamp_local", self.validator.data.columns)


if __name__ == "__main__":
    unittest.main()
