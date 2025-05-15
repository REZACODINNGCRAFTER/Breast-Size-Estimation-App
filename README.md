# Breast Size Estimation App

This is a PyQt5-based GUI application that utilizes machine learning to estimate breast size from uploaded images. The system allows for prediction, feedback logging, and exporting user feedback to Excel for further analysis.

## Features

* **Image Upload**: Users can upload `.jpg`, `.jpeg`, or `.png` images.
* **Prediction**: Uses a trained model to estimate breast size.
* **Feedback Logging**: Users can correct predictions and leave textual feedback.
* **Excel Export**: Feedback can be exported to a structured Excel file.
* **Multi-threaded Prediction**: Keeps the UI responsive during inference.
* **Style and UX**: Clean UI with styled widgets and progress indication.

## Setup Instructions

### Prerequisites

Ensure you have the following installed:

* Python 3.7+
* pip (Python package installer)

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python gui_app.py
```

> **Note**: The GUI assumes there is a module named `breast_size_estimator.py` that contains the `BreastSizeEstimator` class with methods for preparing data and making predictions.

## Feedback Logs

Feedback is stored in the `feedback_logs/` directory as timestamped `.txt` files. These can be exported via the GUI to an Excel file.

## Contributing

We welcome contributions from the community!

### How to Contribute

1. Fork the repository
2. Create a new branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Commit and push (`git commit -am 'Add my feature' && git push origin feature/my-feature`)
5. Open a Pull Request

### Ideas for Contributions

* Improve the machine learning model
* Add support for batch image processing
* Integrate with cloud storage for saving logs
* Implement authentication for feedback management
* Enhance the UI/UX with animations or themes

## License

This project is licensed under the MIT License.

---

If you encounter any issues or have suggestions, feel free to open an issue or email the maintainer.
