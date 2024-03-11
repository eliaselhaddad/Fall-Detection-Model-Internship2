import os
import argparse
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


def process_data(path_to_raw, path_to_processed, s3_helpers_raw, s3_helpers_processed):
    try:
        logger.info("Saving data to S3")
        for file_name in os.listdir(path_to_raw):
            logger.info(f"Uploading {file_name} to S3")
            file_path = os.path.join(path_to_raw, file_name)
            save_data_to_s3 = SaveDataToS3(file_path, s3_helpers_raw)
            save_data_to_s3.save()
    except Exception as e:
        logger.error(f"Error saving data to S3: {e}")

    try:
        logger.info("Saving processed data to S3")
        for file_name in os.listdir(path_to_processed):
            logger.info(f"Uploading {file_name} to S3")
            file_path = os.path.join(path_to_processed, file_name)
            save_data_to_s3 = SaveDataToS3(file_path, s3_helpers_processed)
            save_data_to_s3.save()
    except Exception as e:
        logger.error(f"Error saving processed data to S3: {e}")


def main(folder):
    logger.info("Saving data to S3")

    if folder == "data":
        process_data(
            PATH_TO_RAW,
            PATH_TO_PROCESSED,
            s3_helpers_raw_data,
            s3_helpers_processed_data,
        )
    elif folder == "data2":
        process_data(
            PATH_TO_RAW_2,
            PATH_TO_PROCESSED_2,
            s3_helpers_raw_data_2,
            s3_helpers_processed_data_2,
        )

    logger.info("Data saving process completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload data to S3")
    parser.add_argument("folder", help="Folder to process ('data' or 'data2')")
    args = parser.parse_args()
    main(args.folder)
