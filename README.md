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
    - python -m src.processing.source_all_processor


To process and predict sample data:
- Place data in data/sample_raw
- Use --use_sample tag
    - python -m src.processing.source_all_processor --use_sample
    - python -m src.processing.merge_csv_files --use_sample
    - python -m src.processing.prediction

### For Training Model

- Due to the existence of two data sources the data processing has varied input sources
- First
    - python -m src.processing.source_all_processor
        - will process all the data avalaible from both sources
- Then
    - python -m src.processing.split_sequences
        - This will only split sequences from data source 1. The initial data we had from our data collection
    - python -m src.processing.source_b_preprocessor
        - Will do an additional processing on the data source 2 which is the new data we have collected
- Finally
    - Train the model

- For the training:
    - I have dropped csv files that are longer than 106
    - Some csv files, due to splitting, have less than 70 rows so have all been padded to 110
    - I am using a masking layer in the model to reduce the introduction of bias due to padding




# Flow of events


                    -> merge_csv_files (data1 is default) -> split_sequences (data1 is default)   ->
               data
source_all_processor  ->                                                                                       -> model_training
               data2
                    -> source_b_processor (data2) ->      ->       ->       ->      ->      ->      ->
