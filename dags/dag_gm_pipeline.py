from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from airflow import DAG
import boto3

# from airflow.hooks.S3_hook import S3Hook

# import sys
# from pathlib import Path

# sys.path.append(str(Path("opt/airflow/dags/src").resolve()))

from src.aws.events import get_data_from_s3, save_data_to_s3
from src.processing import data_processor

# def s3_connection(**kwargs):
#     hook = S3Hook(aws_conn_id="gm_s3_conn_id")
#     keys = hook.list_keys(bucket_name="gata-matrix-data")


# @task(task_id="process_data")
# def process_data_task() -> None:
#     return data_processor.main()


# @task(task_id="save_data_to_s3")
# def save_data_to_s3_task() -> None:
#     return save_data_to_s3.main()


# @task(task_id="get_data_from_s3")
# def get_data_from_s3_task() -> None:
#     return get_data_from_s3.main()


# @dag(
#     schedule_interval=timedelta(days=1),
#     start_date=datetime(2023, 2, 9),
#     catchup=False,
#     tags=["gatematrix"],
#     description="DAG for processing data from S3",
#     # python_callable=s3_connection
# )
# def pipeline():
#     (get_data_from_s3_task() >> process_data_task() >> save_data_to_s3_task())


# pipeline_dag = pipeline()

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2023, 2, 9),
}

# add s3 connection


with DAG(
    "gm_pipeline", default_args=default_args, schedule_interval=timedelta(days=1)
) as dag:
    get_data_from_s3_task = PythonOperator(
        task_id="get_data_from_s3",
        python_callable=get_data_from_s3.main,
    )

    process_data_task = PythonOperator(
        task_id="process_data",
        python_callable=data_processor.main,
    )

    save_data_to_s3_task = PythonOperator(
        task_id="save_data_to_s3",
        python_callable=save_data_to_s3.main,
    )

    get_data_from_s3_task >> process_data_task >> save_data_to_s3_task
