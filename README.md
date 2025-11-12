# Breast Size Estimator GUI

The Breast Size Estimator GUI is a professional, PyQt5-based desktop application designed for accurate breast size prediction from images using the `BreastSizeEstimator` model. It offers a modern, user-friendly interface with advanced features such as drag-and-drop image uploads, batch prediction, and comprehensive feedback management. Feedback is stored in a SQLite database, with options to view, filter, and export to Excel. The application supports localization, high-contrast mode for accessibility, and customizable settings via a JSON configuration file, ensuring flexibility and scalability for research and clinical applications. Robust error handling, detailed logging, and a sleek dark theme enhance its reliability and professional appeal.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Directory Structure](#directory-structure)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- **Image Prediction**: Predict breast size from images using the `BreastSizeEstimator` model.
- **Batch Processing**: Upload and predict multiple images simultaneously via drag-and-drop or file dialog.
- **Feedback Management**: Store, view, filter, and export user feedback in a SQLite database.
- **Tabbed Interface**: Separate tabs for prediction and feedback review, with a modern dark theme.
- **Accessibility**: High-contrast mode and screen reader support with accessible widget names.
- **Localization**: Multi-language support using Qt translation files, configurable via JSON.
- **Configurable Settings**: Customize data directory, batch size, and UI preferences through `config.json`.
- **Robust Error Handling**: Retry mechanisms for database and model initialization, with detailed error messages.
- **Status Updates**: Real-time feedback via a status bar and progress bar for predictions.
- **Logging**: Comprehensive logging to `breast_size_app.log` for debugging and auditing.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/breast-size-estimator-gui.git
   cd breast-size-estimator-gui
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Icons and Translations** (optional):
   - Place icon files (`upload.png`, `predict.png`, `save.png`, `export.png`) in the `icons/` directory.
   - Place translation files (e.g., `breast_size_app_fr.qm`) in the `translations/` directory.

5. **Configure the Application**:
   Create a `config.json` file in the root directory (see [Configuration](#configuration)).

## Usage
1. **Run the Application**:
   ```bash
   python breast_size_app.py
   ```

2. **Basic Workflow**:
   - **Upload Images**: Drag and drop images or use the "Upload Images" button (Ctrl+U) to select `.png`, `.jpg`, or `.jpeg` files.
   - **Predict**: Click "Predict" (Ctrl+P) to process all uploaded images.
   - **Provide Feedback**: Enter feedback and select a corrected class (if applicable), then save (Ctrl+S).
   - **View Feedback**: Navigate to the Feedback tab to filter and view stored feedback.
   - **Export Feedback**: Use "Export Feedback" (Ctrl+E) to save feedback to an Excel file.
   - **Exit**: Close the application via File > Exit or Ctrl+Q.

3. **Toggle High-Contrast Mode**:
   - Enable/disable high-contrast mode from the View menu for enhanced accessibility.

## Configuration
The application uses a `config.json` file for customization. If the file is missing, default settings are applied. Example `config.json`:
```json
{
    "data_dir": "path/to/dataset",
    "batch_size": 16,
    "language": "en",
    "high_contrast": false,
    "image_size": 400
}
```
- `data_dir`: Directory containing the dataset for the `BreastSizeEstimator` model.
- `batch_size`: Number of images processed per batch (default: 32).
- `language`: Language code (e.g., `en` for English, `fr` for French).
- `high_contrast`: Enable high-contrast mode (`true`/`false`).
- `image_size`: Size of the image preview area in pixels (default: 400).

## Dependencies
- Python 3.8+
- PyQt5
- pandas
- openpyxl
- opencv-python
- breast_size_estimator (custom module, assumed available)

Install dependencies:
```bash
pip install PyQt5 pandas openpyxl opencv-python
```

## Directory Structure
```
breast-size-estimator-gui/
├── breast_size_app.py       # Main application script
├── config.json              # Configuration file
├── icons/                   # Icon files (upload.png, predict.png, etc.)
├── translations/            # Translation files (breast_size_app_*.qm)
├── breast_size_app.log      # Log file
├── feedback.db              # SQLite database for feedback
└── requirements.txt         # Dependency list
```

## Logging
The application logs events, errors, and user actions to `breast_size_app.log` in the root directory. Logs include timestamps, severity levels, and detailed messages for debugging and auditing.

Example log entry:
```
2025-11-12 20:58:00,123 - INFO - BreastSizeApp initialized at 2025-11-12 20:58:00
2025-11-12 20:58:05,456 - INFO - Uploaded 3 valid images
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request with a detailed description of your changes.

Please ensure code adheres to PEP 8 standards and includes appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions, bug reports, or feature requests, please contact:
- **Email**: support@xai.com
- **GitHub Issues**: [https://github.com/your-repo/breast-size-estimator-gui/issues](https://github.com/your-repo/breast-size-estimator-gui/issues)

---

*Last updated: November 12, 2025* 
