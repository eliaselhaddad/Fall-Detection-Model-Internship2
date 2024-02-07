import os

import boto3
import botocore
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

    def download_file_if_not_local(self, filename: str, local_dir: str) -> None:
        logger.info(f"Checking if file {filename} exists locally")
        local_file_path = os.path.join(local_dir, filename)
        if not os.path.exists(local_file_path):
            logger.info(f"Downloading {filename} from S3")
            self.s3_client.download_file(
                self.bucket, f"{self.prefix}/{filename}", local_file_path
            )
            logger.info(f"Downloaded {filename} to {local_file_path}")
        else:
            logger.info(f"File {filename} already exists locally")

    def upload_file(self, file_path: str, filename: str) -> None:
        logger.info(f"Uploading file {filename} to S3")
        s3_file_path = f"{self.prefix}/{filename}"
        try:
            self.s3_client.upload_file(file_path, self.bucket, s3_file_path)
            logger.info(f"File {filename} uploaded to S3")
        except Exception as e:
            logger.error(f"Error uploading file {filename} to S3: {e}")
            raise e
