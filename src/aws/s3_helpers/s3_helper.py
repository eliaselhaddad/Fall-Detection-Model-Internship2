import os
import tempfile
from typing import Optional

import boto3
import botocore
import pandas as pd
from loguru import logger


class S3Helpers:
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix
        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3")

    def check_file_exists_in_s3_bucket(self, filename: str) -> bool:
        try:
            self.s3_client.head_object(
                Bucket=self.bucket, Key=f"{self.prefix}/{filename}"
            )
            return True
        except botocore.exceptions.ClientError as e:
            return False

    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        logger.info(f"Loading data from {filename}")
        try:
            file_path = f"{self.prefix}/{filename}"
            obj = self.s3_client.get_object(Bucket=self.bucket, Key=file_path)
            return pd.read_csv(obj["Body"])
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {e}")
            raise e

    def save_data_if_not_exists_in_s3(self, data: pd.DataFrame, filename: str) -> None:
        if not self.check_file_exists_in_s3_bucket(filename):
            self.save_data(data, filename)
        else:
            logger.info(f"Data already exists in S3: {filename}")

    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        logger.info(f"Saving data to {filename}")
        file_path = f"{self.prefix}/{filename}"
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            data.to_csv(temp_file.name, index=False)
            self.s3_client.upload_file(temp_file.name, self.bucket, file_path)
        logger.info(f"Data saved to {filename}")

    def get_data_if_does_not_exist_localy(
        self, filename: str
    ) -> Optional[pd.DataFrame]:
        local_file_path = os.path.join(self.prefix, filename)
        if not os.path.exists(local_file_path):
            return self.load_data(filename)
        else:
            return pd.read_csv(local_file_path)

    def upload_file(self, file_path: str, filename: str) -> None:
        logger.info(f"Uploading file {filename} to S3")
        s3_file_path = f"{self.prefix}/{filename}"
        try:
            self.s3_client.upload_file(file_path, self.bucket, s3_file_path)
            logger.info(f"File {filename} uploaded to S3")
        except Exception as e:
            logger.error(f"Error uploading file {filename} to S3: {e}")
            raise e
