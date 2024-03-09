import pandas as pd
from loguru import logger


class DataValidator:
    def __init__(
        self,
    ):
        pass

    def validate(
        self, data: pd.DataFrame, required_accelerator_columns: list
    ) -> pd.DataFrame:
        self._check_all_columns(data, required_accelerator_columns)
        self._check_null_values(data)
        self._check_duplicated_rows()
        self._drop_timestamp()
        self._rename_timestamp_local()
        return self.data

    def _check_all_columns(
        self, required_accelerometer_columns: list, data: pd.DataFrame
    ) -> None:
        missing_columns = set(required_accelerometer_columns) - set(data.columns)
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

    def _check_null_values(self, data: pd.DataFrame) -> None:
        if data.isnull().values.any():
            logger.error("Null values found")
            raise ValueError("Null values found")
        logger.info("No null values found")

    def _check_duplicated_rows(self, data: pd.DataFrame) -> None:
        if data.duplicated().any():
            logger.error("Duplicated rows found")
            raise ValueError("Duplicated rows found")
        logger.info("No duplicated rows found")

    def _drop_timestamp(self, data: pd.DataFrame) -> None:
        if "timestamp" in data.columns:
            data.drop(columns=["timestamp"], inplace=True)
            logger.info("'timestamp' column dropped")
        else:
            logger.warning("'timestamp' column not found; cannot drop")

    def _rename_timestamp_local(self, data: pd.DataFrame) -> None:
        if "timestamp_local" in data.columns:
            data.rename(columns={"timestamp_local": "timestamp"}, inplace=True)
            logger.info("'timestamp_local' column renamed to 'timestamp'")
        else:
            logger.warning("'timestamp_local' column not found; cannot rename")
