import numpy as np
import pandas as pd
from loguru import logger
from scipy.integrate import cumtrapz


class MotionFeatureCalculator:
    def calculate_all_features(
        self,
        dataframe: pd.DataFrame,
        timestamp_col: str,
        accel_cols: list,
        time_unit="ms",
    ) -> pd.DataFrame:
        self.check_all_columns(dataframe, accel_cols)
        self.calculate_time_interval(dataframe, timestamp_col, time_unit)
        self.calculate_velocity_displacement(accel_cols, dataframe)
        self.calculate_angles(dataframe)
        self.calculate_g_force(dataframe)
        self.calculate_jerk_orientation(dataframe)
        self.calculate_magnitudes(dataframe)
        self.calculate_impact_detection(dataframe)
        return self.dataframe

    def check_all_columns(self, dataframe: pd.DataFrame, accel_cols: list) -> None:
        logger.info("Checking if all columns are present")
        for col in accel_cols:
            if col not in dataframe.columns:
                logger.error(f"Column {col} not found in dataframe")
                raise ValueError(f"Column {col} not found in dataframe")
        logger.info("All columns found")

    def calculate_time_interval(self, dataframe, timestamp_col: str, time_unit="ms"):
        logger.info("Calculating time interval")
        dataframe[timestamp_col] = pd.to_datetime(dataframe[timestamp_col])
        dataframe["time_interval"] = (
            dataframe[timestamp_col].diff().fillna(pd.Timedelta(seconds=0))
        )
        dataframe["time_interval"] = dataframe["time_interval"].dt.total_seconds()
        time_divisor = 1000 if time_unit == "ms" else 1
        dataframe["time_interval"] /= time_divisor
        logger.info("Time interval calculated")

    def calculate_velocity_displacement(
        self, accel_cols: list, dataframe: pd.DataFrame
    ):
        logger.info("Calculating velocity and displacement")
        for axis in accel_cols:
            logger.info(f"Calculating velocity and displacement for {axis}")
            velocity_col = f"v{axis[-1]}"
            dataframe[velocity_col] = cumtrapz(
                dataframe[axis], dataframe["time_interval"], initial=0
            )

            displacement_col = f"d{axis[-1]}"
            dataframe[displacement_col] = cumtrapz(
                dataframe[velocity_col], dataframe["time_interval"], initial=0
            )

    def calculate_angles(self, dataframe: pd.DataFrame):
        logger.info("Calculating angles")
        for axis1, axis2 in [("ax", "ay"), ("ay", "az"), ("az", "ax")]:
            logger.info(f"Calculating angle between {axis1} and {axis2}")
            angle_col = f"angle_{axis1[-1]}{axis2[-1]}"
            dataframe[angle_col] = np.degrees(
                np.arctan2(dataframe[axis1], dataframe[axis2])
            )

    def calculate_g_force(self, dataframe: pd.DataFrame):
        logger.info("Calculating g force")
        dataframe["g_force"] = np.sqrt(
            dataframe["ax"] ** 2 + dataframe["ay"] ** 2 + dataframe["az"] ** 2
        )

    def calculate_jerk_orientation(self, dataframe: pd.DataFrame):
        logger.info("Calculating jerk and orientation")
        dataframe["jerk"] = (
            dataframe["g_force"].diff().fillna(0) / dataframe["time_interval"]
        )
        logger.info("Calculating orientation in xy")
        dataframe["orientation_xy"] = np.degrees(
            np.arctan2(dataframe["ay"], dataframe["ax"])
        )

        logger.info("Calculating orientation in yz")
        dataframe["orientation_yz"] = np.degrees(
            np.arctan2(dataframe["az"], dataframe["ay"])
        )

        logger.info("Calculating orientation in zx")
        dataframe["orientation_zx"] = np.degrees(
            np.arctan2(dataframe["ax"], dataframe["az"])
        )

    def calculate_magnitudes(self, dataframe: pd.DataFrame):
        logger.info("Calculating magnitude of acceleration")
        dataframe["magnitude_acceleration"] = np.sqrt(
            dataframe["ax"] ** 2 + dataframe["ay"] ** 2 + dataframe["az"] ** 2
        )

        logger.info("Calculating magnitude of velocity")
        dataframe["magnitude_velocity"] = np.sqrt(
            dataframe["vx"] ** 2 + dataframe["vy"] ** 2 + dataframe["vz"] ** 2
        )

        logger.info("Calculating magnitude of displacement")
        dataframe["magnitude_displacement"] = np.sqrt(
            dataframe["dx"] ** 2 + dataframe["dy"] ** 2 + dataframe["dz"] ** 2
        )

    def calculate_impact_detection(self, dataframe: pd.DataFrame):
        logger.info("Calculating impact detection")
        dataframe["impact_detection"] = dataframe["jerk"].apply(
            lambda x: 1 if x > 75000 else 0
        )
