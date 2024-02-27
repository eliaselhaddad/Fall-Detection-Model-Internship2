import sys
import csv
from datetime import datetime
from enum import Enum

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QGridLayout, QStatusBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QResizeEvent
import numpy as np
import matplotlib.pyplot as plt

import time
# from acc_types import Acceleration
from src.models.Acceleration import Acceleration

"""Annotate Acceleration Data From Accelerometer And Save data as CSV file"""


class SequenceCollectionTypes(Enum):
    no_sequence = None
    non_fall_sequence = 0
    fall_sequence = 1


class SequenceDataCollectionGui(QMainWindow):
    def __init__(self):
        super().__init__()

        self.data_sequence_length = 13 * 6
        self.sequence_recording_type: SequenceCollectionTypes = SequenceCollectionTypes.no_sequence
        self.accelerometer_data: [Acceleration] = []

        self.setWindowTitle("Record Accelerometer Data")
        self.plot_widget = QWidget()
        self.setCentralWidget(self.plot_widget)

        self.layout = QVBoxLayout(self.plot_widget)
        self.figure, self.ax = plt.subplots()
        self.line_x, = self.ax.plot([], [], 'b-', label='X')
        self.line_y, = self.ax.plot([], [], 'g-', label='Y')
        self.line_z, = self.ax.plot([], [], 'r-', label='Z')
        self.ax.set_xlim(0, self.data_sequence_length)
        # Change this according to what max values there are for accelerometer data...
        self.ax.set_ylim(-10, 50)
        self.ax.set_xlabel('Data Points')
        self.ax.set_ylabel('Acceleration')
        self.layout.addWidget(self.figure.canvas)
        self.ax.legend()

        self.x_data = []
        self.y_data = []
        self.z_data = []

        # Buttons and progress bar
        # 13 data points per second, sequence length in seconds = 6


        central_widget = QWidget(self)
        self.layout.addWidget(central_widget)
        layout = QGridLayout(central_widget)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, self.data_sequence_length)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar {margin-top: 10px; border: 1px solid black; border-radius: 12px; color: black;}"
            "QProgressBar::chunk {background-color: #05B8CC; border-radius: 12px;}"
        )

        self.status_bar = QStatusBar(self)
        self.status_bar.setSizeGripEnabled(False)
        layout.addWidget(self.progress_bar, 1, 0, 1, 2)
        layout.addWidget(self.status_bar, 2, 0, 2, 2)

        fall_button = QPushButton("Record fall sequence", self)
        fall_button.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 18px; background-color: green; color: white;}"
        )

        fall_button.adjustSize()
        fall_button.clicked.connect(self.fall_button_clicked)
        layout.addWidget(fall_button, 0, 0)

        not_fall_button = QPushButton("Record non fall sequence", self)
        not_fall_button.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 18px; background-color: red; color: white;}"
        )
        not_fall_button.adjustSize()
        not_fall_button.clicked.connect(self.not_fall_button_clicked)
        layout.addWidget(not_fall_button, 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

    # def ui_plot_component(self):


    def update_plot(self, acceleration: Acceleration = None):
        if acceleration is not None:
            self.x_data.append(acceleration.ax)
            self.y_data.append(acceleration.ay)
            self.z_data.append(acceleration.az)

        self.line_x.set_xdata(range(len(self.x_data)))
        self.line_x.set_ydata(self.x_data)
        self.line_y.set_xdata(range(len(self.y_data)))
        self.line_y.set_ydata(self.y_data)
        self.line_z.set_xdata(range(len(self.z_data)))
        self.line_z.set_ydata(self.z_data)
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.figure.canvas.draw()

    def fall_button_clicked(self):
        self.clear_plot()
        self.sequence_recording_type = SequenceCollectionTypes.fall_sequence
        print(f"Recording {self.sequence_recording_type.name}")
        self.send_fake_data()

    def not_fall_button_clicked(self):
        self.clear_plot()
        self.sequence_recording_type = SequenceCollectionTypes.non_fall_sequence
        print(f"Recording {self.sequence_recording_type.name}")
        self.send_fake_data()

    def stop_sequence_recording(self):
        timestamp = datetime.now().isoformat().replace(":", "-")
        file_name = f"{self.sequence_recording_type.name}_{timestamp}.csv"
        self.save_as_csv(file_name=file_name)
        self.accelerometer_data.clear()
        self.progress_bar.reset()
        self.sequence_recording_type = SequenceCollectionTypes.no_sequence
        self.status_bar.showMessage(f"Saved file {file_name}", 5000)

    def on_data_received(self, acceleration: Acceleration):
        if len(self.accelerometer_data) == self.data_sequence_length:
            print('stop recording')
            self.stop_sequence_recording()

        if self.sequence_recording_type != SequenceCollectionTypes.no_sequence:
            acceleration.fall_state = self.sequence_recording_type.value
            self.accelerometer_data.append(acceleration)
            self.update_plot(acceleration=acceleration)

        self.progress_bar.setValue(len(self.accelerometer_data))

    def clear_plot(self):
        self.x_data.clear()
        self.y_data.clear()
        self.z_data.clear()
        self.update_plot()

    def save_as_csv(self, file_name: str):
        with open(file_name, "w", newline="") as file:
            writer = csv.writer(file)
            head = ["timestamp", "timestamp_local", "ax", "ay", "az", "fall_state"]
            writer.writerow(head)

            for field in self.accelerometer_data:
                writer.writerow(field.as_csv_field())
            file.close()

    # Sending fake data
    def send_fake_data(self):
        print("starting gogo")
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
            acc = Acceleration(1, "2024-02-27 22:00:01", ax, ay, az, '0')
            print(f"for loop iteration: {i}")
            self.on_data_received(acc)
            time.sleep(0.01)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SequenceDataCollectionGui()
    window.show()
