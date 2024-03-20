from datetime import datetime
import os
import re
from typing import List
from loguru import logger

from src.aws.s3_helpers.s3_helper import S3Helpers


class GetDataFromS3:
    def __init__(self):
        self.bucket = "gata-matrix-data"
        self.model_prefix = "model/model"
        self.scaler_prefix = "model/scaler"
        self.local_model_directory = "models/model"
        self.local_scaler_directory = "models/scaler"
        self.s3_helper = S3Helpers(self.bucket, self.model_prefix)

    def get(self) -> None:
        try:
            logger.info("Fetching model data from S3")
            model_contents = self._fetch_bucket_objects(self.model_prefix)
            if model_contents:
                logger.info(f"Found {len(model_contents)} objects in the S3 bucket")
                latest_model_key = self._get_latest_model_key(model_contents)
                self._process_latest_file(latest_model_key, self.local_model_directory)

            logger.info("Fetching scaler data from S3")
            scaler_contents = self._fetch_bucket_objects(f"{self.scaler_prefix}")
            if scaler_contents:
                logger.info(f"Found {len(scaler_contents)} objects in the S3 bucket")
                latest_scaler_key = self._get_latest_scaler_key(scaler_contents)
                self._process_latest_file(
                    latest_scaler_key, self.local_scaler_directory
                )
        except Exception as e:
            logger.error(f"Error in retrieving data from S3: {e}")
            raise e

    def _fetch_bucket_objects(self, prefix: str) -> list:
        try:
            bucket_objects = self.s3_helper.s3_client.list_objects_v2(
                Bucket=self.bucket, Prefix=prefix
            )
            contents = bucket_objects.get("Contents", [])
            if not contents:
                logger.error(
                    f"No objects found in the S3 bucket under the given prefix {prefix}"
                )
            else:
                logger.info(f"Found {len(contents)} objects in the S3 bucket")
            return contents
        except Exception as e:
            logger.exception(f"Error in fetching bucket objects: {e}")
            raise e

    def _get_latest_model_key(self, objects: List[dict]) -> str:
        try:
            date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
            latest_date = None
            latest_file_key = None

            for obj in objects:
                match = date_pattern.search(obj["Key"])
                if match:
                    file_date = datetime.strptime(match.group(), "%Y-%m-%d")
                    if not latest_date or file_date > latest_date:
                        latest_date = file_date
                        latest_file_key = obj["Key"]

            if not latest_file_key:
                raise ValueError("No file found with a date in the key")

            return latest_file_key
        except Exception as e:
            logger.error(f"Error in getting latest file key: {e}")
            raise e

    def _get_latest_scaler_key(self, objects: List[dict]) -> str:
        return objects[-1]["Key"]

    def _process_latest_file(self, file_key: str, local_directory: str) -> None:
        try:
            logger.info(f"Processing latest file {file_key}")
            if local_directory == self.local_model_directory:
                logger.info(f"Processing model file {file_key}")
                date_string = self._extract_date_from_key(file_key)
                local_date_directory = os.path.join(local_directory, date_string)
                os.makedirs(local_date_directory, exist_ok=True)
            else:
                local_date_directory = local_directory

            original_file_name = os.path.basename(file_key)
            local_file_path = os.path.join(local_date_directory, original_file_name)

            if not os.path.exists(local_file_path):
                self._download_file(file_key, local_file_path)
            else:
                logger.info(f"File {local_file_path} already exists locally")
        except Exception as e:
            logger.error(f"Error processing latest file {file_key}: {e}")
            raise e

    def _extract_date_from_key(self, file_key: str) -> str:
        try:
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", file_key)
            if not date_match:
                raise ValueError("No date found in the key")
            logger.info(f"Latest file key: {file_key}")
            return date_match.group()
        except Exception as e:
            logger.error(f"Error extracting date from key {file_key}: {e}")
            raise e

    def _create_local_file_path(self, date_string: str, file_key: str) -> str:
        try:
            local_date_directory = os.path.join(self.local_model_directory, date_string)
            os.makedirs(local_date_directory, exist_ok=True)
            original_file_name = os.path.basename(file_key)
            return os.path.join(local_date_directory, original_file_name)
        except Exception as e:
            logger.error(f"Error creating local file path: {e}")
            raise e

    def _download_file(self, file_key: str, local_file_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            logger.info(f"Downloading {file_key} to {local_file_path}")
            self.s3_helper.s3_client.download_file(
                self.bucket, file_key, local_file_path
            )
            logger.info(f"File downloaded to {local_file_path}")
        except Exception as e:
            logger.error(f"Error downloading file {file_key}: {e}")
            raise e


def main():
    get_data_from_s3 = GetDataFromS3()
    get_data_from_s3.get()


if __name__ == "__main__":
    main()
