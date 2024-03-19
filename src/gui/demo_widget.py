"""python -m src.sensors.movesense_accelerometer"""
from collections import deque
import sys
from pathlib import Path
from winsound import PlaySound, SND_ASYNC
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
)
from PyQt6.QtCore import QTimer, Qt
import numpy as np
import pyqtgraph as pg
from PyQt5.QtMultimedia import QSound
from src.tools.acceleration import Acceleration


class DemoWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Data and settings
        self.data_sequence_length = 13 * 18
        self.g_force_threshold = 12
        self.is_streaming = False

        # Initialize data storage
        self.x_data = deque(maxlen=self.data_sequence_length)
        self.y_data = deque(maxlen=self.data_sequence_length)
        self.z_data = deque(maxlen=self.data_sequence_length)
        self.g_force_data = deque(maxlen=self.data_sequence_length)

        # UI setup
        self.layout = QVBoxLayout(self)
        self.setupPlotWidget()
        self.setupControlButtons()

        # Blink timer and siren sound
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.blink_warning)
        self.siren_sound = QSound("data/sound_effects/siren.wav")

    def setupPlotWidget(self):
        self.plotWidget = pg.PlotWidget()
        self.legend = self.plotWidget.addLegend()
        self.x_plot = self.plotWidget.plot([], pen="r", name="X-axis")
        self.y_plot = self.plotWidget.plot([], pen="g", name="Y-axis")
        self.z_plot = self.plotWidget.plot([], pen="b", name="Z-axis")
        self.g_force_plot = self.plotWidget.plot([], pen="y", name="g_force")
        self.plotWidget.showGrid(x=True, y=True, alpha=0.3)
        self.layout.addWidget(self.plotWidget)

    def setupControlButtons(self):
        # Create a new horizontal layout for the button and the warning label
        bottomLayout = QHBoxLayout()

        # Setup the start/stop streaming button
        self.start_streaming_button = QPushButton("Start/Stop Streaming", self)
        self.start_streaming_button.setMinimumHeight(40)
        self.start_streaming_button.setMinimumWidth(200)
        self.start_streaming_button.setStyleSheet(
            "background-color: #4CAF50; color: white;" "font-size: 20px;"
        )
        self.start_streaming_button.clicked.connect(self.toggleStreaming)
        bottomLayout.addWidget(self.start_streaming_button)

        # Setup the threshold checking button
        self.check_model_button = QPushButton("Threshold Met. Checking...", self)
        self.check_model_button.setMinimumHeight(
            40
        )  # Same height as the start_streaming_button
        self.check_model_button.setMinimumWidth(
            200
        )  # Same width as the start_streaming_button
        self.check_model_button.setStyleSheet(
            "background-color: #2196F3; color: white; font-size: 20px;"  # A different color for distinction
        )
        # Initially disabled, can be enabled based on your application logic
        self.check_model_button.setEnabled(False)
        self.check_model_button.hide()
        bottomLayout.addWidget(self.check_model_button)

        # Setup the warning label
        self.warning_label = QLabel("Fall Detected!", self)
        self.warning_label.setMinimumWidth(600)  # Make the warning label wider
        self.warning_label.setStyleSheet("color: white; background-color: red;")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warning_label.hide()  # Initially hidden
        bottomLayout.addWidget(self.warning_label)

        # A spacer item to push the button and the label to the left
        spacerItem = QSpacerItem(
            20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        bottomLayout.addItem(spacerItem)

        # Horizontal layout with the button, label, and spacer to the main layout
        self.layout.addLayout(bottomLayout)

    def setupWarningLabel(self):
        self.warning_label = QLabel("Fall Detected!", self)
        self.warning_label.setStyleSheet("color: white; background-color: red;")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warning_label.hide()
        self.layout.addWidget(self.warning_label)

    def toggleStreaming(self):
        self.is_streaming = not self.is_streaming
        self.start_streaming_button.setText(
            "Stop Streaming" if self.is_streaming else "Start Streaming"
        )

    def receive_accelerometer_data(self, acceleration: Acceleration):
        if self.is_streaming:
            self.x_data.append(acceleration.ax)
            self.y_data.append(acceleration.ay)
            self.z_data.append(acceleration.az)
            g_force = self.calculate_g_force(acceleration)
            self.g_force_data.append(g_force)
            self.check_g_force_threshold(g_force, self.g_force_threshold)
            self.update_plot()

    def update_plot(self):
        self.x_plot.setData(list(self.x_data))
        self.y_plot.setData(list(self.y_data))
        self.z_plot.setData(list(self.z_data))
        self.g_force_plot.setData(list(self.g_force_data))

    def calculate_g_force(self, acceleration: Acceleration):
        g_force = np.sqrt(
            acceleration.ax**2 + acceleration.ay**2 + acceleration.az**2
        )
        return g_force

    def check_g_force_threshold(self, g_force: float, g_force_threshold: float):
        if g_force > g_force_threshold:
            self.warning_label.show()
            self.blink_timer.start(500)
            self.check_model_button.show()
            # SND_ASYNC flag to play the sound asynchronously without blocking the main thread
            PlaySound("data/sound_effects/siren.wav", SND_ASYNC)

    def blink_warning(self):
        # Toggle the visibility of the warning label to create a blinking effect
        self.warning_label.setVisible(not self.warning_label.isVisible())
