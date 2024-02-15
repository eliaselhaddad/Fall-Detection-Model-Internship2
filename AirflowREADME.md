This readme helps with setting up airflow

- Navigate to root of project
-
    - make sure you are in the root such that docker-compose.yml, Dockerfile and start_airflow are same level
    - RUN: ./start_airflow.sh
    - loggin to aws lia account
    - let the magic happen!

- loggin to airflow:
    - go to: localhost:8080
    - username: airflow
    - password: airflow

- select the gm_pipeline
    - go to graphs
    - start the job
