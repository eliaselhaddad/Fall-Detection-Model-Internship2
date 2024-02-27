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
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QResizeEvent

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
        self.sequence_recording_type: SequenceCollectionTypes = SequenceCollectionTypes.no_sequence
        self.accelerometer_data: [Acceleration] = []
        # 13 data points per second, sequence length in seconds = 6
        self.data_sequence_length = 13 * 6

        self.setWindowTitle("Annotate Accelerometer Data")
        # self.setGeometry(100, 100, 800, 500)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QGridLayout(central_widget)
        self.progress_bar = QProgressBar(self)
        # self.progress_bar.setFixedSize(800, 50)
        # self.progress_bar.move(0, 450)
        self.progress_bar.setRange(0, 78)
        self.status_bar = QStatusBar(self)
        self.status_bar.setSizeGripEnabled(False)
        # self.status_bar.setFixedWidth(500)
        # self.status_bar.move(0, 10)
        # layout.addWidget(self.status_bar, 0, 0)
        layout.addWidget(self.progress_bar, 1, 0, 1, 2)
        layout.addWidget(self.status_bar, 2, 0)

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

    # def resizeEvent(self, event: QResizeEvent) -> None:
    #     width = event.size().width()
    #     height = event.size().height()
    #     self.progress_bar.setFixedWidth(width)

    def fall_button_clicked(self):
        self.sequence_recording_type = SequenceCollectionTypes.fall_sequence
        print(f"Recording {self.sequence_recording_type.name}")
        self.gogo()

    def not_fall_button_clicked(self):
        self.sequence_recording_type = SequenceCollectionTypes.non_fall_sequence
        print(f"Recording {self.sequence_recording_type.name}")
        self.gogo()

    def gogo(self):
        acc = Acceleration(1, "2024-02-27 22:00:01", 0.1, 0.2, 0.3, '0')
        print("starting gogo")
        for i in range(79):
            print(f"for loop iteration: {i}")
            self.on_data_received(acc)
            time.sleep(0.01)

    def stop_sequence_recording(self):
        timestamp = datetime.now().isoformat().replace(":", "-")
        file_name = f"{self.sequence_recording_type.name}_{timestamp}.csv"
        self.save_as_csv(file_name=file_name)
        self.accelerometer_data.clear()
        self.progress_bar.reset()
        self.sequence_recording_type = SequenceCollectionTypes.no_sequence
        self.status_bar.showMessage('Saved file', 5000)

    def on_data_received(self, acceleration: Acceleration):
        if len(self.accelerometer_data) == self.data_sequence_length:
            print('stop recording')
            self.stop_sequence_recording()

        if self.sequence_recording_type != SequenceCollectionTypes.no_sequence:
            acceleration.fall_state = self.sequence_recording_type.value
            self.accelerometer_data.append(acceleration)

        self.progress_bar.setValue(len(self.accelerometer_data))

    def save_as_csv(self, file_name: str):
        with open(file_name, "w", newline="") as file:
            writer = csv.writer(file)
            head = ["timestamp", "timestamp_local", "ax", "ay", "az", "fall_state"]
            writer.writerow(head)

            for field in self.accelerometer_data:
                writer.writerow(field.as_csv_field())
            file.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SequenceDataCollectionGui()
    window.show()
