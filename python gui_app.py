import sys
import os
import json
import sqlite3
from datetime import datetime
import pandas as pd
import logging
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QMessageBox, QTextEdit, QComboBox, QProgressBar, QMenuBar, QAction, QDesktopWidget,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QStatusBar, QLineEdit
)
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QTranslator
from breast_size_estimator import BreastSizeEstimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('breast_size_app.log')
    ]
)
logger = logging.getLogger(__name__)

class PredictThread(QThread):
    """Thread for running breast size predictions in the background."""
    result_ready = pyqtSignal(list)  # List of (image_path, prediction) tuples
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)

    def __init__(self, estimator, image_paths):
        super().__init__()
        self.estimator = estimator
        self.image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

    def run(self):
        try:
            results = []
            total = len(self.image_paths)
            for i, path in enumerate(self.image_paths):
                self.progress_updated.emit(int((i / total) * 100))
                result = self.estimator.predict_breast_size(path)
                results.append((path, result))
            self.progress_updated.emit(100)
            self.result_ready.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))

class BreastSizeApp(QWidget):
    """Main application window for the Breast Size Estimator GUI."""

    def __init__(self):
        super().__init__()
        self.settings = QSettings("xAI", "BreastSizeApp")
        self.translator = QTranslator()
        self.estimator = None
        self.image_paths = []
        self.current_predictions = []
        self.feedback_db = "feedback.db"
        self.config = self._load_config()

        self._init_localization()
        self._init_database()
        self._init_estimator()
        self._init_ui()
        self._center_window()
        logger.info("BreastSizeApp initialized at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _load_config(self) -> dict:
        """Load configuration from a JSON file or defaults."""
        default_config = {
            "data_dir": "dataset",
            "batch_size": 32,
            "language": "en",
            "high_contrast": False,
            "image_size": 400
        }
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r") as f:
                    config = json.load(f)
                default_config.update(config)
                logger.info("Configuration loaded from config.json")
            return default_config
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return default_config

    def _init_localization(self):
        """Initialize localization based on config."""
        lang = self.config.get("language", "en")
        if lang != "en":
            if self.translator.load(f"translations/breast_size_app_{lang}.qm"):
                QApplication.instance().installTranslator(self.translator)
                logger.info("Loaded translation for language: %s", lang)
            else:
                logger.warning("Translation file for %s not found", lang)

    def _init_database(self):
        """Initialize SQLite database with retry mechanism."""
        for attempt in range(3):
            try:
                with sqlite3.connect(self.feedback_db, timeout=10) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS feedback (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT,
                            image_path TEXT,
                            predicted TEXT,
                            corrected TEXT,
                            user_feedback TEXT
                        )
                    """)
                    conn.commit()
                logger.info("Feedback database initialized")
                return
            except sqlite3.Error as e:
                logger.error("Database initialization attempt %d failed: %s", attempt + 1, e)
                if attempt == 2:
                    QMessageBox.critical(self, self.tr("Database Error"), f"Failed to initialize database: {e}")
                    self.close()

    def _init_estimator(self):
        """Initialize the BreastSizeEstimator with retry mechanism."""
        for attempt in range(3):
            try:
                config = {
                    'data_dir': self.config.get("data_dir", "dataset"),
                    'image_size': (224, 224),
                    'batch_size': self.config.get("batch_size", 32)
                }
                self.estimator = BreastSizeEstimator(config)
                self.estimator.prepare_data()
                self.estimator.build_model()
                logger.info("BreastSizeEstimator initialized and model built")
                return
            except Exception as e:
                logger.error("Estimator initialization attempt %d failed: %s", attempt + 1, e)
                if attempt == 2:
                    QMessageBox.critical(self, self.tr("Initialization Error"), f"Failed to initialize model: {e}")
                    self.close()

    def _init_ui(self):
        """Initialize the user interface components."""
        self.setWindowTitle(self.tr("Breast Size Estimator"))
        self.setMinimumSize(900, 700)
        self.setAcceptDrops(True)
        self.setStyleSheet(self._load_styles(high_contrast=self.config.get("high_contrast", False)))

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage(self.tr("Ready"))

        # Menu bar
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu(self.tr("File"))
        exit_action = QAction(self.tr("Exit"), self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu(self.tr("View"))
        high_contrast_action = QAction(self.tr("High Contrast Mode"), self, checkable=True)
        high_contrast_action.setChecked(self.config.get("high_contrast", False))
        high_contrast_action.toggled.connect(self._toggle_high_contrast)
        view_menu.addAction(high_contrast_action)

        # Tabbed interface
        self.tabs = QTabWidget()
        self.prediction_tab = QWidget()
        self.feedback_tab = QWidget()
        self.tabs.addTab(self.prediction_tab, self.tr("Prediction"))
        self.tabs.addTab(self.feedback_tab, self.tr("Feedback"))

        # Prediction tab
        image_size = self.config.get("image_size", 400)
        self.image_label = QLabel(self.tr("Drag or upload images"))
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(image_size, image_size)
        self.image_label.setStyleSheet("border: 2px dashed #ccc; margin-bottom: 10px;")
        self.image_label.setToolTip(self.tr("Drop or upload images for prediction"))
        self.image_label.setAccessibleName("Image display area")

        self.result_label = QLabel(self.tr("Prediction: N/A"))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.result_label.setAccessibleName("Prediction result")

        self.class_selector = QComboBox()
        self.class_selector.addItems(self.estimator.list_available_classes())
        self.class_selector.setEnabled(False)
        self.class_selector.setToolTip(self.tr("Select the correct class if prediction is incorrect"))
        self.class_selector.setAccessibleName("Class selector")

        self.feedback_box = QTextEdit()
        self.feedback_box.setPlaceholderText(self.tr("Optional: Provide feedback about the prediction..."))
        self.feedback_box.setFixedHeight(100)
        self.feedback_box.setToolTip(self.tr("Enter feedback about the prediction"))
        self.feedback_box.setAccessibleName("Feedback input")

        self.upload_button = QPushButton(self.tr("Upload Images"))
        self.upload_button.setIcon(QIcon("icons/upload.png"))
        self.upload_button.clicked.connect(self.upload_images)
        self.upload_button.setToolTip(self.tr("Upload images for prediction (Ctrl+U)"))
        self.upload_button.setShortcut("Ctrl+U")
        self.upload_button.setAccessibleName("Upload images button")

        self.predict_button = QPushButton(self.tr("Predict"))
        self.predict_button.setIcon(QIcon("icons/predict.png"))
        self.predict_button.clicked.connect(self.predict_images)
        self.predict_button.setEnabled(False)
        self.predict_button.setToolTip(self.tr("Run prediction on uploaded images (Ctrl+P)"))
        self.predict_button.setShortcut("Ctrl+P")
        self.predict_button.setAccessibleName("Predict button")

        self.save_feedback_button = QPushButton(self.tr("Save Feedback"))
        self.save_feedback_button.setIcon(QIcon("icons/save.png"))
        self.save_feedback_button.clicked.connect(self.save_feedback)
        self.save_feedback_button.setEnabled(False)
        self.save_feedback_button.setToolTip(self.tr("Save feedback to database (Ctrl+S)"))
        self.save_feedback_button.setShortcut("Ctrl+S")
        self.save_feedback_button.setAccessibleName("Save feedback button")

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAccessibleName("Progress bar")

        prediction_layout = QVBoxLayout()
        prediction_layout.addWidget(self.image_label)
        prediction_layout.addWidget(self.result_label)
        prediction_layout.addWidget(self.class_selector)
        prediction_layout.addWidget(self.feedback_box)
        prediction_layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.predict_button)
        button_layout.addWidget(self.save_feedback_button)
        prediction_layout.addLayout(button_layout)
        self.prediction_tab.setLayout(prediction_layout)

        # Feedback tab
        self.feedback_table = QTableWidget()
        self.feedback_table.setColumnCount(5)
        self.feedback_table.setHorizontalHeaderLabels([
            self.tr("Timestamp"), self.tr("Image Path"), self.tr("Predicted"),
            self.tr("Corrected"), self.tr("Feedback")
        ])
        self.feedback_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.feedback_table.setAccessibleName("Feedback table")

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText(self.tr("Filter by image path or feedback..."))
        self.filter_input.textChanged.connect(self._filter_feedback)
        self.filter_input.setToolTip(self.tr("Enter text to filter feedback entries"))
        self.filter_input.setAccessibleName("Feedback filter input")

        self.export_excel_button = QPushButton(self.tr("Export Feedback"))
        self.export_excel_button.setIcon(QIcon("icons/export.png"))
        self.export_excel_button.clicked.connect(self.export_feedback_to_excel)
        self.export_excel_button.setToolTip(self.tr("Export feedback to Excel (Ctrl+E)"))
        self.export_excel_button.setShortcut("Ctrl+E")
        self.export_excel_button.setAccessibleName("Export feedback button")

        feedback_layout = QVBoxLayout()
        feedback_layout.addWidget(self.filter_input)
        feedback_layout.addWidget(self.feedback_table)
        feedback_layout.addWidget(self.export_excel_button)
        self.feedback_tab.setLayout(feedback_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(menubar)
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(self.status_bar)
        self.setLayout(main_layout)

        self._load_feedback()

    def _load_styles(self, high_contrast: bool = False) -> str:
        """Load and return the stylesheet for the application."""
        base_style = """
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLabel {
                font-size: 16px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #1e90ff;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #4682b4;
            }
            QPushButton:disabled {
                background-color: #696969;
            }
            QTextEdit, QComboBox, QLineEdit {
                font-size: 14px;
                background-color: #353535;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 5px;
                text-align: center;
                color: #ffffff;
                background-color: #353535;
            }
            QProgressBar::chunk {
                background-color: #1e90ff;
            }
            QMenuBar {
                background-color: #353535;
                color: #ffffff;
            }
            QMenuBar::item {
                background-color: #353535;
                color: #ffffff;
            }
            QMenuBar::item:selected {
                background-color: #4682b4;
            }
            QTableWidget {
                background-color: #353535;
                color: #ffffff;
                gridline-color: #555555;
            }
            QHeaderView::section {
                background-color: #353535;
                color: #ffffff;
                border: 1px solid #555555;
            }
        """
        if high_contrast:
            base_style += """
                QWidget {
                    background-color: #000000;
                    color: #ffffff;
                }
                QLabel, QTextEdit, QComboBox, QLineEdit {
                    background-color: #000000;
                    color: #ffffff;
                    border: 2px solid #ffffff;
                }
                QPushButton {
                    background-color: #00b7eb;
                    color: #ffffff;
                    border: 2px solid #ffffff;
                }
                QTableWidget {
                    background-color: #000000;
                    color: #ffffff;
                    gridline-color: #ffffff;
                }
            """
        return base_style

    def _toggle_high_contrast(self, checked: bool):
        """Toggle high contrast mode."""
        self.config["high_contrast"] = checked
        self.setStyleSheet(self._load_styles(high_contrast=checked))
        self.status_bar.showMessage(self.tr("High contrast mode ") + ("enabled" if checked else "disabled"))
        logger.info("High contrast mode %s", "enabled" if checked else "disabled")

    def _center_window(self):
        """Center the window on the screen."""
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def dragEnterEvent(self, event):
        """Handle drag enter events for image files."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle drop events for image files."""
        urls = event.mimeData().urls()
        valid_images = [url.toLocalFile() for url in urls if self._validate_image(url.toLocalFile())]
        if valid_images:
            self.image_paths = valid_images
            self._display_image(valid_images[0])
            self.predict_button.setEnabled(True)
            self.save_feedback_button.setEnabled(False)
            self.feedback_box.clear()
            self.class_selector.setEnabled(False)
            self.result_label.setText(self.tr("Prediction: N/A"))
            self.status_bar.showMessage(self.tr(f"{len(valid_images)} image(s) loaded"))
            logger.info("Dropped %d valid images", len(valid_images))
        else:
            QMessageBox.warning(self, self.tr("Invalid Images"), self.tr("Please drop valid image files (.png, .jpg, .jpeg)"))

    def upload_images(self):
        """Handle multiple image uploads and display the first one."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, self.tr("Select Images"), "", self.tr("Image Files (*.png *.jpg *.jpeg)")
        )
        if file_paths:
            valid_images = [path for path in file_paths if self._validate_image(path)]
            if valid_images:
                self.image_paths = valid_images
                self._display_image(valid_images[0])
                self.predict_button.setEnabled(True)
                self.save_feedback_button.setEnabled(False)
                self.feedback_box.clear()
                self.class_selector.setEnabled(False)
                self.result_label.setText(self.tr("Prediction: N/A"))
                self.status_bar.showMessage(self.tr(f"{len(valid_images)} image(s) loaded"))
                logger.info("Uploaded %d valid images", len(valid_images))
            else:
                QMessageBox.warning(self, self.tr("Invalid Images"), self.tr("Please upload valid image files"))

    def _display_image(self, file_path: str):
        """Display the specified image in the image label."""
        pixmap = QPixmap(file_path).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio
        )
        self.image_label.setPixmap(pixmap)

    def _validate_image(self, file_path: str) -> bool:
        """Validate if the file is a valid image."""
        valid_extensions = {'.png', '.jpg', '.jpeg'}
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in valid_extensions:
            return False
        try:
            img = cv2.imread(file_path)
            return img is not None
        except Exception:
            return False

    def predict_images(self):
        """Initiate batch prediction process in a separate thread."""
        if self.image_paths:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.predict_button.setEnabled(False)
            self.thread = PredictThread(self.estimator, self.image_paths)
            self.thread.result_ready.connect(self.handle_prediction_results)
            self.thread.error_occurred.connect(self.handle_prediction_error)
            self.thread.progress_updated.connect(self.progress_bar.setValue)
            self.thread.start()
            self.status_bar.showMessage(self.tr("Predicting..."))
            logger.info("Batch prediction started for %d images", len(self.image_paths))
        else:
            QMessageBox.warning(self, self.tr("No Images"), self.tr("Please upload or drop images first"))
            logger.warning("Prediction attempted without images")

    def handle_prediction_results(self, results: list):
        """Handle successful batch prediction results."""
        self.current_predictions = results
        if results:
            first_path, first_result = results[0]
            self._display_image(first_path)
            self.result_label.setText(self.tr(f"Prediction (first image): {first_result}"))
            self.save_feedback_button.setEnabled(True)
            self.class_selector.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.predict_button.setEnabled(True)
        self.status_bar.showMessage(self.tr(f"Prediction completed for {len(results)} image(s)"))
        logger.info("Batch prediction completed: %d results", len(results))
        QMessageBox.information(
            self, self.tr("Prediction Complete"),
            self.tr(f"Predicted {len(results)} image(s). First prediction: {first_result if results else 'N/A'}")
        )

    def handle_prediction_error(self, error: str):
        """Handle prediction errors."""
        QMessageBox.critical(self, self.tr("Prediction Error"), self.tr(f"Failed to predict: {error}"))
        self.progress_bar.setVisible(False)
        self.predict_button.setEnabled(True)
        self.status_bar.showMessage(self.tr("Prediction failed"))
        logger.error("Prediction error: %s", error)

    def save_feedback(self):
        """Save user feedback to the SQLite database for all predicted images."""
        if not self.current_predictions:
            QMessageBox.warning(self, self.tr("No Predictions"), self.tr("Please make predictions before saving feedback"))
            return

        feedback = self.feedback_box.toPlainText().strip()
        if not feedback:
            QMessageBox.warning(self, self.tr("Empty Feedback"), self.tr("Please provide feedback before saving"))
            return

        try:
            selected_class = self.class_selector.currentText()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with sqlite3.connect(self.feedback_db, timeout=10) as conn:
                cursor = conn.cursor()
                for image_path, predicted in self.current_predictions:
                    cursor.execute(
                        """
                        INSERT INTO feedback (timestamp, image_path, predicted, corrected, user_feedback)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (timestamp, image_path, predicted, selected_class, feedback)
                    )
                conn.commit()
            self.feedback_box.clear()
            self.class_selector.setEnabled(False)
            self.save_feedback_button.setEnabled(False)
            self.current_predictions = []
            self._load_feedback()
            self.status_bar.showMessage(self.tr("Feedback saved"))
            logger.info("Feedback saved for %d images", len(self.current_predictions))
            QMessageBox.information(self, self.tr("Feedback Saved"), self.tr("Your feedback has been recorded"))
        except sqlite3.Error as e:
            logger.error("Failed to save feedback: %s", e)
            QMessageBox.critical(self, self.tr("Database Error"), self.tr(f"Failed to save feedback: {e}"))

    def export_feedback_to_excel(self):
        """Export feedback from the database to an Excel file."""
        try:
            with sqlite3.connect(self.feedback_db, timeout=10) as conn:
                query = "SELECT * FROM feedback"
                df = pd.read_sql_query(query, conn)
            if df.empty:
                QMessageBox.warning(self, self.tr("No Data"), self.tr("No feedback data available to export"))
                return
            export_path, _ = QFileDialog.getSaveFileName(
                self, self.tr("Save Excel File"), "feedback_summary.xlsx", self.tr("Excel Files (*.xlsx)")
            )
            if export_path:
                df.to_excel(export_path, index=False)
                self.status_bar.showMessage(self.tr(f"Feedback exported to {export_path}"))
                logger.info("Feedback exported to: %s", export_path)
                QMessageBox.information(
                    self, self.tr("Export Complete"), self.tr(f"Feedback exported to {export_path}")
                )
        except Exception as e:
            logger.error("Failed to export feedback: %s", e)
            QMessageBox.critical(self, self.tr("Export Error"), self.tr(f"Failed to export feedback: {e}"))

    def _load_feedback(self):
        """Load feedback from the database into the feedback table."""
        try:
            with sqlite3.connect(self.feedback_db, timeout=10) as conn:
                query = "SELECT * FROM feedback ORDER BY timestamp DESC"
                df = pd.read_sql_query(query, conn)
            self.feedback_table.setRowCount(len(df))
            for row_idx, row in df.iterrows():
                self.feedback_table.setItem(row_idx, 0, QTableWidgetItem(row['timestamp']))
                self.feedback_table.setItem(row_idx, 1, QTableWidgetItem(row['image_path']))
                self.feedback_table.setItem(row_idx, 2, QTableWidgetItem(row['predicted']))
                self.feedback_table.setItem(row_idx, 3, QTableWidgetItem(row['corrected']))
                self.feedback_table.setItem(row_idx, 4, QTableWidgetItem(row['user_feedback']))
            logger.info("Feedback table loaded with %d entries", len(df))
        except sqlite3.Error as e:
            logger.error("Failed to load feedback: %s", e)
            QMessageBox.critical(self, self.tr("Database Error"), self.tr(f"Failed to load feedback: {e}"))

    def _filter_feedback(self, text: str):
        """Filter feedback table based on search text."""
        try:
            with sqlite3.connect(self.feedback_db, timeout=10) as conn:
                query = """
                    SELECT * FROM feedback
                    WHERE image_path LIKE ? OR user_feedback LIKE ?
                    ORDER BY timestamp DESC
                """
                df = pd.read_sql_query(query, conn, params=(f"%{text}%", f"%{text}%"))
            self.feedback_table.setRowCount(len(df))
            for row_idx, row in df.iterrows():
                self.feedback_table.setItem(row_idx, 0, QTableWidgetItem(row['timestamp']))
                self.feedback_table.setItem(row_idx, 1, QTableWidgetItem(row['image_path']))
                self.feedback_table.setItem(row_idx, 2, QTableWidgetItem(row['predicted']))
                self.feedback_table.setItem(row_idx, 3, QTableWidgetItem(row['corrected']))
                self.feedback_table.setItem(row_idx, 4, QTableWidgetItem(row['user_feedback']))
            logger.info("Feedback table filtered with %d entries for query: %s", len(df), text)
        except sqlite3.Error as e:
            logger.error("Failed to filter feedback: %s", e)
            QMessageBox.critical(self, self.tr("Database Error"), self.tr(f"Failed to filter feedback: {e}"))

    def closeEvent(self, event):
        """Handle window close event."""
        reply = QMessageBox.question(
            self, self.tr("Exit"), self.tr("Are you sure you want to exit?"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            logger.info("Application closed by user at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = BreastSizeApp()
    window.show()
    sys.exit(app.exec_())
