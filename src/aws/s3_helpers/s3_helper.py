from datetime import datetime
import os
import re

import boto3
import botocore
from loguru import logger


class S3Helpers:
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix
        self.aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )
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

    def get_latest_date(self, path) -> datetime:
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        latest_date = None

        for filename in os.listdir(path):
            match = date_pattern.search(filename)
            if match:
                date = datetime.strptime(match.group(), "%Y-%m-%d")
                if latest_date is None or date > latest_date:
                    latest_date = date

        if latest_date is None:
            raise ValueError("No date found")
        return latest_date
