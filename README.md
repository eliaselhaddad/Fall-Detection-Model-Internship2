# Fall Detection Model

## Introduction
This project focuses on detecting falls using sensor data. The model processes data from various sources, including BLE sensors, and uses machine learning techniques to identify fall events accurately.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [References](#references)

## Project Structure
- **dags/**: Directed Acyclic Graphs for orchestrating tasks.
- **hyperparameters/**: Hyperparameter configurations for model training.
- **src/**: Source code for data processing and model training.
- **tests/**: Unit tests for the project.
- **.coveragerc**: Configuration file for measuring code coverage.
- **.gitignore**: Specifies intentionally untracked files to ignore.
- **.isort.cfg**: Configuration for sorting imports.
- **.pre-commit-config.yaml**: Configuration for pre-commit hooks.
- **AirflowREADME.md**: Instructions specific to Airflow setup.
- **Dockerfile**: Docker configuration file.
- **Makefile**: Commands for setting up and managing the project environment.
- **README.md**: Project overview and instructions.
- **cdk.json, cdk_json.txt**: AWS CDK configuration files.
- **docker-compose.yml**: Docker Compose configuration.
- **pass-role-policy.json**: AWS policy for passing roles.
- **poetry.lock**: Lock file for Poetry dependencies.
- **postgres.env**: Environment variables for PostgreSQL.
- **pyproject.toml**: Configuration file for project dependencies.
- **requirements.txt**: List of project dependencies.
- **start_airflow.sh**: Script to start Airflow.
- **trust-policy.json**: AWS trust policy configuration.

## Setup Instructions

### Prerequisites
- **Python 3.10 or later**
- **AWS CLI**: Ensure that your AWS CLI is configured with the necessary credentials.
- **Node.js and npm**: Required for AWS CDK.
- **Docker**: For containerized deployment.

### Steps
1. **Clone the repository**:
    ```sh
    git clone https://github.com/eliaselhaddad/Fall-Detection-Model-Internship2.git
    cd Fall-Detection-Model-Internship2
    ```

2. **Set up the virtual environment** (optional but recommended):
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install dependencies using Poetry**:
    ```sh
    pip install poetry
    poetry install
    ```

4. **Set up AWS CDK**:
    ```sh
    npm install -g aws-cdk
    cdk bootstrap
    ```

5. **Set up Docker**:
    ```sh
    docker-compose up
    ```

## How It Works
The Fall Detection Model processes sensor data from BLE sensors, performs data preprocessing, and trains a machine learning model to detect falls. The workflow involves the following steps:
- **Data Retrieval**: Collects data from BLE sensors.
- **Data Processing**: Cleans and processes the data for model training.
- **Model Training**: Trains a machine learning model using the processed data.
- **Event Handling**: Uses AWS services like Lambda and CDK for managing events and infrastructure.

### Example Commands
- **Run the application**:
    ```sh
    python -m src.app
    ```

- **Process and predict sample data**:
    ```sh
    python -m src.processing.source_all_processor --use_sample
    ```

## References
- [AWS CDK](https://docs.aws.amazon.com/cdk/latest/guide/home.html)
- [Poetry](https://python-poetry.org/docs/)
- [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [Pandas](https://pandas.pydata.org/)
