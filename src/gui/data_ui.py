import sys
import csv
from dataclasses import asdict
from datetime import datetime
from enum import Enum
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QStatusBar,
    QTabWidget,
)

import numpy as np
import matplotlib.pyplot as plt

import time
from src.gui.demo_widget import DemoWidget
from src.tools.acceleration import Acceleration
from src.gui.sequence_data_collection_widget import SequenceDataCollectionWidget

"""Annotate Acceleration Data From Accelerometer And Save data as CSV file"""


class DataUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DataUI")
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.data_collection_widget = SequenceDataCollectionWidget()
        self.demo_widget = DemoWidget()
        self.tabs.addTab(self.data_collection_widget, "Data Collection")
        self.tabs.addTab(self.demo_widget, "Demo")
        self.tabs.tabBarClicked.connect(self.tab_clicked)
        self.chosen_tab_index = self.tabs.currentIndex()

    def tab_clicked(self, index):
        self.chosen_tab_index = index
        print(self.chosen_tab_index)

    def on_data_received(self, acceleration: Acceleration):
        if self.chosen_tab_index == 0:
            self.data_collection_widget.receive_accelerometer_data(acceleration)
        elif self.chosen_tab_index == 1:
            self.demo_widget.receive_accelerometer_data(acceleration)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataUI()
    window.show()
    sys.exit(app.exec())
