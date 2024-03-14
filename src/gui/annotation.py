import sys
import csv

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QGridLayout,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

from datetime import datetime

# from acc_types import Acceleration
from src.tools.acceleration import Acceleration


from src.modelling.model_trigger import ModelTrigger

"""Annotate Acceleration Data From Accelerometer And Save data as CSV file"""

DATA_POINTS = []


class AnnotateAccelerometerData(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Annotate Accelerometer Data")
        self.setGeometry(100, 100, 800, 500)
        self.fall_state = "0"

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QGridLayout(central_widget)

        fall_button = QPushButton("Start Fall", self)
        fall_button.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 18px; background-color: green; color: white;}"
        )
        fall_button.setFixedSize(200, 100)
        fall_button.clicked.connect(self.fall_button_clicked)
        layout.addWidget(fall_button, 0, 0)

        not_fall_button = QPushButton("Stop Fall", self)
        not_fall_button.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 18px; background-color: red; color: white;}"
        )
        not_fall_button.setFixedSize(200, 100)
        not_fall_button.clicked.connect(self.not_fall_button_clicked)
        layout.addWidget(not_fall_button, 0, 1)

        pause_button = QPushButton("Pause", self)
        pause_button.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 18px; background-color: orange; color: white;}"
        )
        pause_button.setFixedSize(200, 100)
        pause_button.clicked.connect(self.pause_button_clicked)
        layout.addWidget(pause_button, 1, 0)

        restart_button = QPushButton("Restart", self)
        restart_button.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 18px; background-color: purple; color: white;}"
        )
        restart_button.setFixedSize(200, 100)
        restart_button.clicked.connect(self.restart_button_clicked)
        layout.addWidget(restart_button, 1, 1)

        save_button = QPushButton("Save As CSV", self)
        save_button.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 18px; background-color: gray; color: white;}"
        )
        save_button.setFixedSize(200, 100)
        save_button.clicked.connect(self.save_as_csv)
        layout.addWidget(save_button, 2, 0)

        self.model_trigger = ModelTrigger(csv_row_limit=78, csv_overlap=26)

    def fall_button_clicked(self):
        self.fall_state = "Start"
        print("Fall", self.fall_state)

    def not_fall_button_clicked(self):
        self.fall_state = "Stop"
        print("Not Fall", self.fall_state)

    def pause_button_clicked(self):
        self.fall_state = "Pause"
        print("Pause", self.fall_state)

    def restart_button_clicked(self):
        self.fall_state = "Restart"
        print("Restart", self.fall_state)

    def on_data_received(self, acceleration: Acceleration):
        # print(acceleration)
        self.model_trigger.update_data_window(acceleration)

        if self.fall_state == "Start":
            acceleration.fall_state = "1"
        elif self.fall_state == "Stop":
            acceleration.fall_state = "0"
        elif self.fall_state == "Pause":
            acceleration.fall_state = "2"
        elif self.fall_state == "Restart":
            acceleration.fall_state = "3"
        else:
            acceleration.fall_state = "0"
        DATA_POINTS.append(acceleration)

    def save_as_csv(self):
        with open(f"sensor_data_{datetime.now()}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            head = ["timestamp", "timestamp_local", "ax", "ay", "az", "fall_state"]
            writer.writerow(head)

            for field in DATA_POINTS:
                writer.writerow(field.as_csv_field())
        DATA_POINTS.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnnotateAccelerometerData()
    window.show()
