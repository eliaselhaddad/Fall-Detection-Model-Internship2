from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from airflow import DAG
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

from src.aws.events import get_data_from_s3, save_data_to_s3
from src.processing import data_processor

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "start_date": datetime(2024, 1, 1),
}


with DAG(
    "gm_pipeline",
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    description="DAG for processing data from S3",
    catchup=False,
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

    process_data_task >> get_data_from_s3_task >> save_data_to_s3_task
