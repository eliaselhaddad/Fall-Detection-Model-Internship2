"""python -m src.sensors.movesense_accelerometer"""

from collections import deque
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QGridLayout,
)
from PyQt6.QtCore import (
    QTimer,
    Qt,
    QUrl,
    QObject,
    QThread,
    pyqtSignal,
    QMetaObject,
    Q_ARG,
    pyqtSlot,
)
from loguru import logger
from PyQt6.QtMultimedia import QSoundEffect
import numpy as np
import pyqtgraph as pg

from src.tools.acceleration import Acceleration
from src.modelling.model_trigger import ModelTrigger


class ModelWorker(QObject):
    updateStatusSignal = pyqtSignal(str)
    updatePredictionSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.model_trigger = ModelTrigger(csv_row_limit=78, csv_overlap=26)

    @pyqtSlot(Acceleration)
    def run_model_check(self, acceleration):
        try:
            self.model_trigger.update_data_window(acceleration)
            should_model_trigger = self.model_trigger.get_latest_trigger_status()

            if should_model_trigger:
                self.updateStatusSignal.emit(should_model_trigger)

            if self.model_trigger.should_display_fall():
                self.updatePredictionSignal.emit(
                    self.model_trigger.get_latest_prediction()
                )
            else:
                self.model_trigger.predict_conclusion = None
        except Exception as e:
            logger.error(f"Error in model worker: {e}")


class DemoWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Data and settings
        self.data_sequence_length = 13 * 18
        self.g_force_threshold = 12
        self.is_streaming = False
        self.max_last_fall_items = 1

        # Initialize data storage
        self.x_data = deque(maxlen=self.data_sequence_length)
        self.y_data = deque(maxlen=self.data_sequence_length)
        self.z_data = deque(maxlen=self.data_sequence_length)
        self.g_force_data = deque(maxlen=self.data_sequence_length)
        self.last_fall_data = deque(maxlen=self.max_last_fall_items)

        # UI setup
        self.layout = QVBoxLayout(self)
        self.setupPlotWidget()
        self.setupControlButtons()

        # Blink timer and siren sound
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.blink_warning)
        self.stop_blink_timer = QTimer(self)
        self.stop_blink_timer.timeout.connect(self.stop_blinking)

        # Sound set up
        alert_path = "assets/alert.wav"
        self.siren_sound = QSoundEffect(self)
        self.siren_sound.setSource(QUrl.fromLocalFile(alert_path))
        self.siren_sound.setVolume(1.0)

        # Thread manager setup
        self.thread = QThread()
        self.worker = ModelWorker()
        self.worker.moveToThread(self.thread)

        self.worker.updateStatusSignal.connect(self.update_model_trigger_status)
        self.worker.updatePredictionSignal.connect(self.update_prediction_status)
        self.thread.start()

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
        gridLayout = QGridLayout()

        gridLayout.setContentsMargins(20, 20, 20, 20)
        gridLayout.setSpacing(20)

        self.createStreamingButton(gridLayout, 0, 0)
        self.createLastFallView(gridLayout, 0, 1)
        self.createTriggerInfoView(gridLayout, 1, 0)
        self.createFallInfoView(gridLayout, 1, 1)

        self.layout.addLayout(gridLayout)

    def createStreamingButton(self, layout, row, column):
        self.start_streaming_button = QPushButton("Start/Stop Streaming", self)
        self.start_streaming_button.setFixedSize(300, 100)
        self.start_streaming_button.setStyleSheet(
            "background-color: #24788F; color: white; font-size: 24px; border-radius: 8px; font-weight: bold; font-family: Monospace;"
        )
        self.start_streaming_button.clicked.connect(self.toggleStreaming)
        layout.addWidget(self.start_streaming_button, row, column)

    def createLastFallView(self, layout, row, column):
        self.last_fall_label = QLabel("No Fall Recorded Yet", self)

        self.last_fall_label.setFixedSize(800, 100)
        self.last_fall_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.last_fall_label.setStyleSheet(
            "background-color: #849DAB; color: white; font-size: 24px; border-radius: 8px;  font-weight: bold; font-family: Monospace;"
        )
        layout.addWidget(self.last_fall_label, row, column)

    def updateLastFallView(self):
        if not self.last_fall_data:
            self.last_fall_label.setText("No Fall Recorded Yet")
        else:
            fall_times = [
                "Last fall recorded at: " + time.strftime("%Y-%m-%d %H:%M:%S")
                for time in self.last_fall_data
            ]
            self.last_fall_label.setText("\n".join(fall_times))

    def record_fall(self):
        self.last_fall_data.appendleft(datetime.now())
        self.updateLastFallView()

    def createTriggerInfoView(self, layout, row, column):
        self.trigger_info_label = QLabel("Trigger Status: Ok", self)
        self.trigger_info_label.setFixedSize(300, 100)
        self.trigger_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.updateTriggerInfoView("Ok")
        layout.addWidget(self.trigger_info_label, row, column)

    def updateTriggerInfoView(self, status):
        status_text = "Trigger Status: " + status
        self.trigger_info_label.setText(status_text)
        if status == "Triggered":
            self.trigger_info_label.setStyleSheet(
                "background-color: #FFBF00; color: white; font-size: 22px; border-radius: 8px;  font-weight: bold; font-family: Monospace;"
            )
        else:
            self.trigger_info_label.setStyleSheet(
                "background-color: #FFFFFF; color: black; font-size: 22px; border-radius: 8px; font-weight: bold; font-family: Monospace;"
            )

    def createFallInfoView(self, layout, row, column):
        self.fall_info_label = QLabel("Fall Status Ok", self)
        self.fall_info_label.setFixedSize(800, 100)
        self.fall_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.updateFallInfoView("Ok")
        layout.addWidget(self.fall_info_label, row, column)

    def updateFallInfoView(self, status):
        status_text = "Fall Status: " + status + " Detected"
        self.fall_info_label.setText(status_text)
        if status == "Fall":
            self.fall_info_label.setStyleSheet(
                "background-color: red; color: white; font-size: 24px; border-radius: 8px; font-weight: bold; font-family: Monospace;"
            )
        else:
            self.fall_info_label.setStyleSheet(
                "background-color: #FFFFFF; color: black; font-size: 24px; border-radius: 8px; font-weight: bold; font-family: Monospace;"
            )

    def resetFall(self):
        self.fall_info_label.setText("Fall Status Ok")
        self.fall_info_label.setStyleSheet(
            "background-color: #FFFFFF; color: black; font-size: 24px; border-radius: 8px;font-weight: bold; font-family: Monospace;"
        )

    def toggleStreaming(self):
        self.is_streaming = not self.is_streaming
        self.start_streaming_button.setText(
            "Stop Streaming" if self.is_streaming else "Start Streaming"
        )

    def receive_accelerometer_data(self, acceleration: Acceleration):
        self.update_data(acceleration)
        self.update_plot()
        QMetaObject.invokeMethod(
            self.worker,
            "run_model_check",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(Acceleration, acceleration),
        )

    def update_data(self, acceleration: Acceleration):
        self.x_data.append(acceleration.ax)
        self.y_data.append(acceleration.ay)
        self.z_data.append(acceleration.az)
        g_force = self.calculate_g_force(acceleration)
        self.g_force_data.append(g_force)

    def update_model_trigger_status(self, status: str):
        self.threshold_trigger_status = status
        self.trigger_info_label.setText(status)
        self.updateTriggerInfoView(status)
        self.trigger_info_label.show()
        self.trigger_info_label.setEnabled(True)

    def update_prediction_status(self, prediction: str):
        self.prediction_status = prediction
        self.fall_info_label.setText(prediction)
        self.updateFallInfoView(prediction)
        self.fall_info_label.show()

        if prediction == "Fall":
            self.blink_timer.start(500)
            self.stop_blink_timer.start(5000)
            self.record_fall()

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

    def blink_warning(self):
        self.fall_info_label.setVisible(not self.fall_info_label.isVisible())

        if not self.fall_info_label.isVisible():
            self.siren_sound.play()

    def stop_blinking(self):
        self.blink_timer.stop()
        self.fall_info_label.setVisible(True)
        self.resetFall()
        self.siren_sound.stop()
