import os
from loguru import logger

from src.aws.s3_helpers.s3_helper import S3Helpers


class SaveModelToS3:
    def __init__(self):
        self.model_directory = "models/model"
        self.scaler_directory = "models/scaler"
        self.bucket = "gata-matrix-data"
        self.model_prefix = "model"
        self.scaler_prefix = "scaler"
        self.s3_helper = S3Helpers(self.bucket, self.model_prefix)

    def save(self) -> None:
        try:
            latest_date = self.s3_helper.get_latest_date(self.model_directory)
            formatted_date = latest_date.strftime("%Y-%m-%d")

            local_model_path = self._get_path(
                self.model_directory, "fall_detection_model.keras", formatted_date
            )
            self._upload_file(local_model_path, self.model_prefix, formatted_date)

            local_scaler_path = self._get_path(self.scaler_directory, "scaler.pkl", "")
            self._upload_file(local_scaler_path, self.scaler_prefix, "")
        except Exception as e:
            logger.error(f"Error saving to S3: {e}")
            raise e

    def _get_path(self, base_directory: str, filename: str, formatted_date: str) -> str:
        logger.info("Getting path for file")
        path = (
            os.path.join(base_directory, formatted_date, filename)
            if formatted_date
            else os.path.join(base_directory, filename)
        )
        return path

    def _upload_file(self, local_path: str, prefix: str, formatted_date: str) -> None:
        filename = os.path.basename(local_path)
        file_path_to_s3 = (
            f"{prefix}/{formatted_date}/{filename}"
            if formatted_date
            else f"{prefix}/{filename}"
        )

        logger.info(f"Checking if {file_path_to_s3} exists in S3")
        if not self.s3_helper.check_file_exists_in_s3_bucket(file_path_to_s3):
            logger.info(f"Uploading {filename} to S3")
            self.s3_helper.upload_file(local_path, file_path_to_s3)
        else:
            logger.info(f"{file_path_to_s3} already exists in S3")


def main():
    save_model_to_s3 = SaveModelToS3()
    save_model_to_s3.save()


if __name__ == "__main__":
    main()
