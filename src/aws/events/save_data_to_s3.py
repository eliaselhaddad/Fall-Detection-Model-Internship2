import os
from loguru import logger

from src.aws.s3_helpers.s3_helper import S3Helpers

BUCKET_NAME = "gata-matrix-data"
RAW_DATA_PREFIX = "raw_data"
PROCESSED_DATA_PREFIX = "processed_data"
PATH_TO_RAW = "data/raw"
PATH_TO_PROCESSED = "data/processed"
s3_helpers_raw_data = S3Helpers(BUCKET_NAME, RAW_DATA_PREFIX)
s3_helpers_processed_data = S3Helpers(BUCKET_NAME, PROCESSED_DATA_PREFIX)


class SaveDataToS3:
    def __init__(self, data_path, s3_helper: S3Helpers):
        self.data_path = data_path
        self.s3_helper = s3_helper

    def save(self):
        filename = os.path.basename(self.data_path)
        if not self.s3_helper.check_file_exists_in_s3_bucket(filename):
            logger.info(f"Uploading {filename} to S3")
            self.s3_helper.upload_file(self.data_path, filename)
        else:
            logger.info(f"{filename} already exists in S3")


def main():
    logger.info("Saving data to S3")

    for file_name in os.listdir(PATH_TO_RAW):
        file_path = os.path.join(PATH_TO_RAW, file_name)
        save_data_to_s3 = SaveDataToS3(file_path, s3_helpers_raw_data)
        save_data_to_s3.save()

    for file_name in os.listdir(PATH_TO_PROCESSED):
        file_path = os.path.join(PATH_TO_PROCESSED, file_name)
        save_data_to_s3 = SaveDataToS3(file_path, s3_helpers_processed_data)
        save_data_to_s3.save()

    logger.info("Data saving process completed successfully")


if __name__ == "__main__":
    main()
