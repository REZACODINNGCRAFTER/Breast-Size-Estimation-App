import sys
import os
from datetime import datetime
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QFileDialog,
                             QVBoxLayout, QHBoxLayout, QMessageBox, QTextEdit, QComboBox, QProgressBar)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from breast_size_estimator import BreastSizeEstimator

class PredictThread(QThread):
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, estimator, image_path):
        super().__init__()
        self.estimator = estimator
        self.image_path = image_path

    def run(self):
        try:
            result = self.estimator.predict_breast_size(self.image_path)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

class BreastSizeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Breast Size Estimator")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(self._load_styles())

        self.estimator = BreastSizeEstimator()
        self._init_model()
        self._init_ui()

        self.image_path = None
        self.current_prediction = None

    def _load_styles(self):
        return """
            QLabel {
                font-size: 16px;
            }
            QPushButton {
                background-color: #5cacee;
                color: white;
                padding: 6px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #a9a9a9;
            }
            QTextEdit, QComboBox {
                font-size: 14px;
            }
        """

    def _init_model(self):
        try:
            self.estimator.prepare_data()
            self.estimator.build_model()
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", str(e))

    def _init_ui(self):
        self.image_label = QLabel("Upload an image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(300, 300)
        self.image_label.setStyleSheet("border: 2px dashed #ccc; margin-bottom: 10px;")

        self.result_label = QLabel("Prediction: N/A")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))

        self.class_selector = QComboBox()
        self.class_selector.addItems(self.estimator.list_available_classes())
        self.class_selector.setEnabled(False)

        self.feedback_box = QTextEdit()
        self.feedback_box.setPlaceholderText("Optional: Provide feedback about the prediction...")

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict_image)
        self.predict_button.setEnabled(False)

        self.save_feedback_button = QPushButton("Save Feedback")
        self.save_feedback_button.clicked.connect(self.save_feedback)
        self.save_feedback_button.setEnabled(False)

        self.export_excel_button = QPushButton("Export Feedback to Excel")
        self.export_excel_button.clicked.connect(self.export_feedback_to_excel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.class_selector)
        layout.addWidget(self.feedback_box)
        layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.predict_button)
        button_layout.addWidget(self.save_feedback_button)
        button_layout.addWidget(self.export_excel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.result_label.setText("Prediction: N/A")
            self.predict_button.setEnabled(True)
            self.save_feedback_button.setEnabled(False)
            self.feedback_box.clear()
            QMessageBox.information(self, "Image Loaded", "Image loaded successfully. Ready to predict.")

    def predict_image(self):
        if self.image_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.predict_button.setEnabled(False)
            self.thread = PredictThread(self.estimator, self.image_path)
            self.thread.result_ready.connect(self.handle_prediction_result)
            self.thread.error_occurred.connect(self.handle_prediction_error)
            self.thread.start()
        else:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")

    def handle_prediction_result(self, result):
        self.current_prediction = result
        self.result_label.setText(f"Prediction: {result}")
        self.save_feedback_button.setEnabled(True)
        self.class_selector.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.predict_button.setEnabled(True)
        QMessageBox.information(self, "Prediction Complete", f"The predicted breast size is: {result}")

    def handle_prediction_error(self, error):
        QMessageBox.critical(self, "Prediction Error", f"Failed to predict: {error}")
        self.progress_bar.setVisible(False)
        self.predict_button.setEnabled(True)

    def save_feedback(self):
        if self.image_path and self.current_prediction:
            selected_class = self.class_selector.currentText()
            feedback = self.feedback_box.toPlainText()

            feedback_dir = "feedback_logs"
            os.makedirs(feedback_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(feedback_dir, f"{base_name}_{timestamp}.txt")

            with open(log_path, 'w') as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Image: {self.image_path}\n")
                f.write(f"Predicted: {self.current_prediction}\n")
                f.write(f"Corrected: {selected_class}\n")
                f.write(f"User Feedback: {feedback}\n")

            QMessageBox.information(self, "Feedback Saved", "Your feedback has been recorded.")
            self.feedback_box.clear()
            self.class_selector.setEnabled(False)
            self.save_feedback_button.setEnabled(False)

    def export_feedback_to_excel(self):
        feedback_dir = "feedback_logs"
        if not os.path.exists(feedback_dir):
            QMessageBox.warning(self, "No Data", "No feedback logs available to export.")
            return

        records = []
        for file in os.listdir(feedback_dir):
            if file.endswith(".txt"):
                path = os.path.join(feedback_dir, file)
                with open(path, 'r') as f:
                    data = f.read().splitlines()
                    record = {}
                    for line in data:
                        if ": " in line:
                            key, value = line.split(": ", 1)
                            record[key] = value
                    record["Filename"] = file
                    records.append(record)

        if records:
            df = pd.DataFrame(records)
            required_columns = ['Filename', 'Timestamp', 'Image', 'Predicted', 'Corrected', 'User Feedback']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''
            df = df[required_columns]
            df.sort_values(by='Timestamp', ascending=False, inplace=True)
            export_path, _ = QFileDialog.getSaveFileName(self, "Save Excel File", "feedback_summary.xlsx", "Excel Files (*.xlsx)")
            if export_path:
                df.to_excel(export_path, index=False)
                QMessageBox.information(self, "Export Complete", f"Feedback has been exported to {export_path}")
        else:
            QMessageBox.warning(self, "Empty Data", "No valid feedback data found to export.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BreastSizeApp()
    window.show()
    sys.exit(app.exec_())
