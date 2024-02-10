from datetime import datetime, timedelta

from airflow.decorators import dag, task

# import sys
# from pathlib import Path

# sys.path.append(str(Path("opt/airflow/dags/src").resolve()))

from src.aws.events import get_data_from_s3, save_data_to_s3
from src.processing import data_processor


@task(task_id="process_data")
def process_data_task() -> None:
    return data_processor.main()


@task(task_id="save_data_to_s3")
def save_data_to_s3_task() -> None:
    return save_data_to_s3.main()


@task(task_id="get_data_from_s3")
def get_data_from_s3_task() -> None:
    return get_data_from_s3.main()


@dag(
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 2, 9),
    catchup=False,
    tags=["gatematrix"],
    description="DAG for processing data from S3",
)
def pipeline():
    (get_data_from_s3_task() >> process_data_task() >> save_data_to_s3_task())


pipeline_dag = pipeline()
