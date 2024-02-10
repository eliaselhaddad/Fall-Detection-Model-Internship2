FROM apache/airflow:2.8.1-python3.10

# Switch to root to install any necessary system packages
USER root
RUN apt-get update && \
    apt-get install -y \
    # Add any system dependencies you might need here. Example:
    # build-essential \
    # libpq-dev \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt

ENV AIRFLOW__CORE__FERNET_KEY=ulrALwJ4qlgx4GCP1dpj1Zx7ZoKXQsS1mvF_iQOoib4=
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
