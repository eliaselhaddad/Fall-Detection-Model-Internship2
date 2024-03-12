import csv
from dataclasses import asdict
from datetime import datetime, time
from enum import Enum

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget,
    QGridLayout,
    QProgressBar,
    QStatusBar,
    QPushButton,
    QVBoxLayout,
)
from matplotlib import pyplot as plt

import time
from src.tools.acceleration import Acceleration

"""Annotate Acceleration Data From Accelerometer And Save data as CSV file"""


class SequenceCollectionTypes(Enum):
    no_sequence = None
    non_fall_sequence = 0
    fall_sequence = 1


class SequenceDataCollectionWidget(QWidget):
    def __init__(self):
        super().__init__()

        # This is what decides how many data points to collect before saving to csv
        self.data_sequence_length = 13 * 6
        self.sequence_recording_type: SequenceCollectionTypes = (
            SequenceCollectionTypes.no_sequence
        )
        self.accelerometer_data: [Acceleration] = []

        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.jerk_data = [[]]
        self.line_x = None
        self.line_y = None
        self.line_z = None
        self.jerk_line = None
        self.figure, self.ax = plt.subplots()
        # self.layout: QVBoxLayout = self.create_plot_widget()

        self.button_widget = QWidget(self)
        button_layout = QGridLayout(self.button_widget)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, self.data_sequence_length)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar {margin-top: 10px; border: 1px solid black; border-radius: 12px; color: black;}"
            "QProgressBar::chunk {background-color: #05B8CC; border-radius: 12px;}"
        )

        self.status_bar = QStatusBar(self)
        self.status_bar.setSizeGripEnabled(False)
        button_layout.addWidget(self.progress_bar, 1, 0, 1, 2)
        button_layout.addWidget(self.status_bar, 2, 0, 2, 2)

        self.fall_button = QPushButton("Record fall sequence", self)
        self.fall_button.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 18px; background-color: green; color: white;}"
        )

        self.fall_button.adjustSize()
        # fall_button.clicked(self.fall_button_clicked)
        self.fall_button.clicked.connect(self.fall_button_clicked)
        button_layout.addWidget(self.fall_button, 0, 0)

        self.not_fall_button = QPushButton("Record non fall sequence", self)
        self.not_fall_button.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 18px; background-color: red; color: white;}"
        )
        self.not_fall_button.adjustSize()
        self.not_fall_button.clicked.connect(self.not_fall_button_clicked)
        button_layout.addWidget(self.not_fall_button, 0, 1)
        button_layout.setColumnStretch(0, 1)
        button_layout.setColumnStretch(1, 1)

        # self.layout.addWidget(self.button_widget)

    def receive_accelerometer_data(self, acceleration: Acceleration):
        if len(self.accelerometer_data) == self.data_sequence_length:
            print("stop recording")
            self.stop_sequence_recording()

        if self.sequence_recording_type != SequenceCollectionTypes.no_sequence:
            acceleration.fall_state = self.sequence_recording_type.value
            self.accelerometer_data.append(acceleration)
            self.update_plot(acceleration=acceleration)

        self.progress_bar.setValue(len(self.accelerometer_data))

    def create_plot_widget(self):
        plot_widget = QWidget()
        self.setCentralWidget(plot_widget)
        layout = QVBoxLayout(plot_widget)
        self.figure, self.ax = plt.subplots()
        (self.line_x,) = self.ax.plot([], [], "b-", label="X")
        (self.line_y,) = self.ax.plot([], [], "g-", label="Y")
        (self.line_z,) = self.ax.plot([], [], "r-", label="Z")
        self.ax.set_xlim(0, self.data_sequence_length)
        # Change this according to what max values there are for accelerometer data...
        self.ax.set_ylim(-100, 100)
        self.ax.set_xlabel("Data Points")
        self.ax.set_ylabel("Acceleration")

        ax2 = self.ax.twinx()
        ax2.set_ylim(-1000000, 1000000)
        ax2.set_ylabel("Jerk")
        (self.jerk_line,) = ax2.plot([], [], "orange", label="Jerk")

        layout.addWidget(self.figure.canvas)
        self.ax.legend()
        return layout

    def update_plot(self, acceleration: Acceleration = None):
        if acceleration is not None:
            self.x_data.append(acceleration.ax)
            self.y_data.append(acceleration.ay)
            self.z_data.append(acceleration.az)

        if len(self.accelerometer_data) > 77:
            self.jerk_data.append(self.calculate_jerk(self.accelerometer_data))
            self.jerk_line.set_xdata(range(len(self.jerk_data[0])))
            self.jerk_line.set_ydata(self.jerk_data[0])

        self.line_x.set_xdata(range(len(self.x_data)))
        self.line_x.set_ydata(self.x_data)
        self.line_y.set_xdata(range(len(self.y_data)))
        self.line_y.set_ydata(self.y_data)
        self.line_z.set_xdata(range(len(self.z_data)))
        self.line_z.set_ydata(self.z_data)
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.figure.canvas.draw()

    @staticmethod
    def calculate_jerk(accelerations: [Acceleration]) -> [float]:
        if accelerations is not None and len(accelerations) > 1:
            dicts = [asdict(acceleration) for acceleration in accelerations]
            df = pd.DataFrame(dicts)
            df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
            df["time_interval"] = (
                df["timestamp_local"].diff().fillna(pd.Timedelta(seconds=0))
            )
            df["time_interval"] = df["time_interval"].dt.total_seconds()
            time_divisor = 1000
            df["time_interval"] /= time_divisor
            df["g_force"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
            df["jerk"] = df["g_force"].diff().fillna(9.8) / df["time_interval"]
            # Print the max value of the jerk column
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["jerk"], how="all", inplace=True)
            print(f"Max jerk value: {df['jerk'].max()}")
            print(f"Min jerk value: {df['jerk'].min()}")

            return df["jerk"].tolist()

    def fall_button_clicked(self):
        self.clear_plot()
        self.sequence_recording_type = SequenceCollectionTypes.fall_sequence
        print(f"Recording {self.sequence_recording_type.name}")
        self.fall_button.setEnabled(False)

    def not_fall_button_clicked(self):
        self.clear_plot()
        self.sequence_recording_type = SequenceCollectionTypes.non_fall_sequence
        print(f"Recording {self.sequence_recording_type.name}")
        self.not_fall_button.setEnabled(False)

    def stop_sequence_recording(self):
        timestamp = datetime.now().isoformat().replace(":", "-")
        file_name = f"{self.sequence_recording_type.name}_{timestamp}.csv"
        self.save_as_csv(file_name=file_name)
        self.accelerometer_data.clear()
        self.progress_bar.reset()
        self.sequence_recording_type = SequenceCollectionTypes.no_sequence
        self.status_bar.showMessage(f"Saved file {file_name}", 5000)
        self.fall_button.setEnabled(True)
        self.not_fall_button.setEnabled(True)

    def clear_plot(self):
        self.x_data.clear()
        self.y_data.clear()
        self.z_data.clear()
        self.jerk_data.clear()
        self.jerk_line.set_xdata(range(len(self.jerk_data)))
        self.jerk_line.set_ydata(self.jerk_data)
        self.update_plot()

    def save_as_csv(self, file_name: str):
        with open(file_name, "w", newline="") as file:
            writer = csv.writer(file)
            head = ["timestamp", "timestamp_local", "ax", "ay", "az", "fall_state"]
            writer.writerow(head)

            for field in self.accelerometer_data:
                writer.writerow(field.as_csv_field())
            file.close()

    def send_fake_data(self):
        timestamp = datetime.now()
        x = 0
        y = 0
        z = 0
        for i in range(79):
            if 20 < i < 25:
                x += 8.1
                y += 6.1
                z += 2.5
            if 26 < i < 28:
                x -= 33
                y -= 23
                z -= 10
            ax, ay, az = np.random.uniform(-2, 2, size=3)
            ax += x
            ay += y
            az += z
            timestamp += pd.Timedelta(milliseconds=78)
            time_added = timestamp.isoformat()

            acc = Acceleration(1, time_added, ax, ay, az, "0")
            print(f"for loop iteration: {i}")
            self.on_data_received(acc)
            time.sleep(0.01)
