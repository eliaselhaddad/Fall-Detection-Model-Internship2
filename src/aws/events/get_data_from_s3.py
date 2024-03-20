import argparse
import os
from loguru import logger

from src.aws.s3_helpers.s3_helper import S3Helpers

BUCKET_NAME = "gata-matrix-data"
RAW_DATA_PREFIX = "raw_data"
RAW_DATA_2_PREFIX = "raw_data_2"
PROCESSED_DATA_PREFIX = "processed_data"
PROCESSED_DATA_2_PREFIX = "processed_data_2"
PATH_TO_RAW = "data/raw"
PATH_TO_RAW_2 = "data2/raw"
PATH_TO_PROCESSED = "data/processed"
PATH_TO_PROCESSED_2 = "data2/processed"

s3_helpers_raw_data = S3Helpers(BUCKET_NAME, RAW_DATA_PREFIX)
s3_helpers_raw_data_2 = S3Helpers(BUCKET_NAME, RAW_DATA_2_PREFIX)
s3_helpers_processed_data = S3Helpers(BUCKET_NAME, PROCESSED_DATA_PREFIX)
s3_helpers_processed_data_2 = S3Helpers(BUCKET_NAME, PROCESSED_DATA_2_PREFIX)


class GetDataFromS3:
    def __init__(self, s3_helper: S3Helpers, local_directory: str):
        self.s3_helper = s3_helper
        self.local_directory = local_directory

    def get(self):
        bucket_objects = self.s3_helper.s3_client.list_objects_v2(
            Bucket=BUCKET_NAME, Prefix=self.s3_helper.prefix
        )
        for obj in bucket_objects.get("Contents", []):
            logger.info(f"Downloading {obj['Key']}")
            file_name = os.path.basename(obj["Key"])
            self.s3_helper.download_file_if_not_local(file_name, self.local_directory)


def orchestrate_data_fetching(data_folder):
    if data_folder == "data":
        logger.info(f"Fetching data from data folder raw_data and processed_data")
        fetch_data(s3_helpers_raw_data, PATH_TO_RAW)
        fetch_data(s3_helpers_processed_data, PATH_TO_PROCESSED)
    elif data_folder == "data2":
        logger.info("Fetching data from data2 folder raw_data_2 and processed_data_2")
        fetch_data(s3_helpers_raw_data_2, PATH_TO_RAW_2)
        fetch_data(s3_helpers_processed_data_2, PATH_TO_PROCESSED_2)


def fetch_data(s3_helper, local_directory):
    logger.info(
        f"Fetching raw data from S3 to {local_directory} from s3 bucket {s3_helper.prefix}"
    )
    data_fetcher = GetDataFromS3(s3_helper, local_directory)
    data_fetcher.get()


def main():
    parser = argparse.ArgumentParser(description="Fetch data from S3")
    parser.add_argument(
        "data_folder",
        choices=["data", "data2"],
        help="The folder to fetch data from. Options: data, data2",
    )
    args = parser.parse_args()
    orchestrate_data_fetching(args.data_folder)


if __name__ == "__main__":
    main()
