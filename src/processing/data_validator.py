import pandas as pd
from loguru import logger


class DataValidator:
    def __init__(
        self,
        data: pd.DataFrame,
        required_accelerator_columns: list,
    ):
        self.data = data
        self.required_accelerometer_columns = required_accelerator_columns

    def validate(self) -> pd.DataFrame:
        self._check_all_columns()
        self._check_null_values()
        self._check_duplicated_rows()
        self._drop_timestamp()
        self._rename_timestamp_local()
        return self.data

    def _check_all_columns(self) -> None:
        missing_columns = set(self.required_accelerometer_columns) - set(
            self.data.columns
        )
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

    def _check_null_values(self) -> None:
        if self.data.isnull().values.any():
            logger.error("Null values found")
            raise ValueError("Null values found")
        logger.info("No null values found")

    def _check_duplicated_rows(self) -> None:
        if self.data.duplicated().any():
            logger.error("Duplicated rows found")
            raise ValueError("Duplicated rows found")
        logger.info("No duplicated rows found")

    def _drop_timestamp(self) -> None:
        if "timestamp" in self.data.columns:
            self.data.drop(columns=["timestamp"], inplace=True)
            logger.info("'timestamp' column dropped")
        else:
            logger.warning("'timestamp' column not found; cannot drop")

    def _rename_timestamp_local(self) -> None:
        if "timestamp_local" in self.data.columns:
            self.data.rename(columns={"timestamp_local": "timestamp"}, inplace=True)
            logger.info("'timestamp_local' column renamed to 'timestamp'")
        else:
            logger.warning("'timestamp_local' column not found; cannot rename")
