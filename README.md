# BLE Sensor data retrieval

The script will configure the native BLE sensor in central configuration and scan for
the sensor ending with given SENSOR_ID. Once a new sensor is found then initiates the
connection, set the command characteristics values and enable notification service.
All the binary data packet will be parsed and sensor payload objects will be stored
locally and save into a csv file on the  ctrl+c exit signal.

Install Bleak before running the script by

    pip install -r requirements.txt


Usage:


    python3 -m src/movesensor.py

##### Starting project

Run
```shell
    make bootstrap
```
##### Running scripts

Due to complications with python modules:
 - run for example:
    - python -m src.processing.data_processor


To process and predict sample data:
- Place data in data/sample_raw
- Use --use_sample tag
    - python -m src.processing.data_processor --use_sample
    - python -m src.processing.merge_csv_files --use_sample
    - python -m src.processing.prediction
