import os

# from dotenv import load_dotenv
from loguru import logger

from src.aws.s3_helpers.s3_helper import S3Helpers

# load_dotenv()
# BUCKET_NAME = os.getenv("BUCKET_NAME")
# RAW_DATA_PREFIX = os.getenv("RAW_DATA_PREFIX")
# PROCESSED_DATA_PREFIX = os.getenv("PROCESSED_DATA_PREFIX")
# PATH_TO_RAW = os.getenv("PATH_TO_RAW")
# PATH_TO_PROCESSED = os.getenv("PATH_TO_PROCESSED")
BUCKET_NAME = "gata-matrix-data"
RAW_DATA_PREFIX = "raw_data"
PROCESSED_DATA_PREFIX = "processed_data"
PATH_TO_RAW = "data/raw"
PATH_TO_PROCESSED = "data/processed"

s3_helpers_raw_data = S3Helpers(BUCKET_NAME, RAW_DATA_PREFIX)
s3_helpers_processed_data = S3Helpers(BUCKET_NAME, PROCESSED_DATA_PREFIX)


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


def main():
    logger.info("Fetching data from S3")
    logger.info("Fetching raw data from S3")
    get_raw_data_from_s3 = GetDataFromS3(s3_helpers_raw_data, PATH_TO_RAW)
    get_raw_data_from_s3.get()

    logger.info("Fetching processed data from S3")
    get_processed_data_from_s3 = GetDataFromS3(
        s3_helpers_processed_data, PATH_TO_PROCESSED
    )
    get_processed_data_from_s3.get()

    logger.info("Data fetching process completed successfully")


if __name__ == "__main__":
    main()
