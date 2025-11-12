import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import keras_tuner as kt
from tqdm import tqdm
import imgaug.augmenters as iaa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('breast_size_estimator.log')
    ]
)
logger = logging.getLogger(__name__)

class BreastSizeEstimator:
    """A class for estimating breast size from images using deep learning models."""

    DEFAULT_CONFIG = {
        'image_size': (224, 224),
        'batch_size': 32,
        'epochs': 20,
        'data_dir': 'dataset',
        'use_transfer_learning': True,
        'base_model': 'MobileNetV2',  # Options: MobileNetV2, EfficientNetB0
        'optimizer': {
            'type': 'Adam',
            'learning_rate': 0.001
        },
        'dropout_rate': 0.4,
        'dense_units': 128,
        'augmentation_params': {
            'rotation_range': 25,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.1,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        },
        'use_advanced_augmentation': False,
        'k_folds': 5,
        'checkpoint_path': 'best_model.h5'
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the BreastSizeEstimator with a configuration dictionary.

        Args:
            config (Dict, optional): Configuration parameters. Defaults to DEFAULT_CONFIG.

        Raises:
            ValueError: If required configuration parameters are invalid.

        Example:
            config = {
                'image_size': (224, 224),
                'batch_size': 16,
                'base_model': 'EfficientNetB0',
                'optimizer': {'type': 'RMSprop', 'learning_rate': 0.0001}
            }
            estimator = BreastSizeEstimator(config)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self._validate_config()

        self.image_size = self.config['image_size']
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['epochs']
        self.data_dir = self.config['data_dir']
        self.use_transfer_learning = self.config['use_transfer_learning']
        self.base_model_name = self.config['base_model']

        self.model: Optional[Model] = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.history = None
        logger.info("BreastSizeEstimator initialized with config: %s", self.config)

        # Enable mixed precision training for performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not os.path.isdir(self.config['data_dir']):
            raise ValueError(f"Data directory '{self.config['data_dir']}' does not exist.")
        if not isinstance(self.config['image_size'], tuple) or len(self.config['image_size']) != 2:
            raise ValueError("image_size must be a tuple of (height, width).")
        if self.config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive.")
        if self.config['epochs'] <= 0:
            raise ValueError("epochs must be positive.")
        if self.config['base_model'] not in ['MobileNetV2', 'EfficientNetB0']:
            raise ValueError("base_model must be 'MobileNetV2' or 'EfficientNetB0'.")
        if self.config['optimizer']['type'] not in ['Adam', 'RMSprop', 'SGD']:
            raise ValueError("optimizer.type must be 'Adam', 'RMSprop', or 'SGD'.")

    def _get_advanced_augmentation(self) -> iaa.Sequential:
        """Define advanced augmentation pipeline using imgaug."""
        return iaa.Sequential([
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Sometimes(0.3, iaa.ContrastNormalization((0.75, 1.25))),
            iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))),
            iaa.Sometimes(0.3, iaa.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            ))
        ])

    def prepare_data(self, test_split: float = 0.1) -> None:
        """
        Prepare training, validation, and test data generators.

        Args:
            test_split (float): Fraction of data to use for testing (default: 0.1).
        """
        try:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                **self.config['augmentation_params']
            )
            test_datagen = ImageDataGenerator(rescale=1./255)

            if self.config['use_advanced_augmentation']:
                train_datagen.preprocessing_function = self._get_advanced_augmentation().augment_image

            self.train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )

            self.val_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )

            # Prepare test data if test_split is provided
            if test_split > 0:
                self.test_generator = test_datagen.flow_from_directory(
                    self.data_dir,
                    target_size=self.image_size,
                    batch_size=self.batch_size,
                    class_mode='categorical',
                    shuffle=False
                )
            logger.info("Data generators prepared: %d training, %d validation, %d test samples",
                        self.train_generator.samples, self.val_generator.samples,
                        self.test_generator.samples if self.test_generator else 0)
        except Exception as e:
            logger.error("Failed to prepare data generators: %s", e)
            raise

    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Return the configured optimizer."""
        opt_config = self.config['optimizer']
        opt_type = opt_config['type']
        lr = opt_config['learning_rate']

        if opt_type == 'Adam':
            return tf.keras.optimizers.Adam(learning_rate=lr)
        elif opt_type == 'RMSprop':
            return tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif opt_type == 'SGD':
            return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

    def build_model(self) -> None:
        """Build and compile the neural network model."""
        try:
            if self.use_transfer_learning:
                self.model = self._build_transfer_learning_model()
            else:
                self.model = self._build_custom_cnn_model()

            self.model.compile(
                optimizer=self._get_optimizer(),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("Model built and compiled with %s.", self.config['base_model'])
        except Exception as e:
            logger.error("Failed to build model: %s", e)
            raise

    def _build_transfer_learning_model(self) -> Model:
        """Build a transfer learning model using the specified base model."""
        base_model_class = MobileNetV2 if self.base_model_name == 'MobileNetV2' else EfficientNetB0
        base_model = base_model_class(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.image_size, 3)
        )
        base_model.trainable = False

        inputs = Input(shape=(*self.image_size, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.config['dense_units'], activation='relu')(x)
        x = Dropout(self.config['dropout_rate'])(x)
        outputs = Dense(self.train_generator.num_classes, activation='softmax')(x)
        return Model(inputs, outputs)

    def _build_custom_cnn_model(self) -> Sequential:
        """Build a custom CNN model."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(self.config['dense_units'], activation='relu'),
            Dropout(self.config['dropout_rate']),
            Dense(self.train_generator.num_classes, activation='softmax')
        ])
        return model

    def hyperparameter_tuning(self) -> None:
        """Perform hyperparameter tuning using Keras Tuner."""
        def model_builder(hp):
            base_model = MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_shape=(*self.image_size, 3)
            )
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(
                hp.Int('units', min_value=64, max_value=256, step=64),
                activation='relu'
            )(x)
            x = Dropout(
                hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
            )(x)
            predictions = Dense(self.train_generator.num_classes, activation='softmax')(x)

            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(
                optimizer=self._get_optimizer(),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model

        try:
            tuner = kt.Hyperband(
                model_builder,
                objective='val_accuracy',
                max_epochs=10,
                factor=3,
                directory='kt_tuning',
                project_name='breast_size_tuning'
            )

            stop_early = EarlyStopping(monitor='val_loss', patience=3)
            tuner.search(
                self.train_generator,
                validation_data=self.val_generator,
                epochs=10,
                callbacks=[stop_early]
            )

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            self.model = tuner.hypermodel.build(best_hps)
            logger.info("Hyperparameter tuning completed with best parameters: %s", best_hps.values)
        except Exception as e:
            logger.error("Hyperparameter tuning failed: %s", e)
            raise

    def train_model(self, use_checkpoint: bool = True) -> Dict:
        """
        Train the model and return training history.

        Args:
            use_checkpoint (bool): If True, save the best model during training.

        Returns:
            Dict: Training history.
        """
        try:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
            ]
            if use_checkpoint:
                callbacks.append(ModelCheckpoint(
                    self.config['checkpoint_path'],
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ))

            self.history = self.model.fit(
                self.train_generator,
                epochs=self.epochs,
                validation_data=self.val_generator,
                callbacks=callbacks,
                verbose=0  # Use tqdm for progress
            )
            logger.info("Model training completed.")
            return self.history.history
        except Exception as e:
            logger.error("Model training failed: %s", e)
            raise

    def fine_tune_model(self) -> None:
        """Fine-tune the transfer learning model."""
        if not self.use_transfer_learning or not self.model:
            logger.warning("Fine-tuning is only supported for transfer learning models.")
            return

        try:
            self.model.layers[0].trainable = True
            for layer in self.model.layers[0].layers[:100]:
                layer.trainable = False
            for layer in self.model.layers[0].layers[100:]:
                layer.trainable = True

            self.model.compile(
                optimizer=self._get_optimizer(),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            self.model.fit(
                self.train_generator,
                epochs=self.epochs // 2,
                validation_data=self.val_generator,
                verbose=0  # Use tqdm for progress
            )
            logger.info("Model fine-tuning completed.")
        except Exception as e:
            logger.error("Model fine-tuning failed: %s", e)
            raise

    def save_model(self, file_path: str = "breast_size_classifier.h5") -> None:
        """Save the trained model to a file."""
        try:
            self.model.save(file_path)
            logger.info("Model saved to %s", file_path)
        except Exception as e:
            logger.error("Failed to save model: %s", e)
            raise

    def load_model(self, file_path: str = "breast_size_classifier.h5") -> None:
        """Load a model from a file."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file '{file_path}' not found.")
            self.model = load_model(file_path)
            logger.info("Model loaded from %s", file_path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    def predict_breast_size(self, image_path: str) -> str:
        """
        Predict the breast size from an image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            str: Predicted class label.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the model or generators are not initialized.
        """
        if not self.model or not self.train_generator:
            raise ValueError("Model and data generators must be initialized before prediction.")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found.")

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            img = cv2.resize(img, self.image_size)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            predictions = self.model.predict(img, verbose=0)
            class_index = np.argmax(predictions[0])
            class_labels = list(self.train_generator.class_indices.keys())
            predicted_label = class_labels[class_index]
            logger.info("Predicted breast size: %s for image %s", predicted_label, image_path)
            return predicted_label
        except Exception as e:
            logger.error("Prediction failed for image %s: %s", image_path, e)
            raise

    def evaluate_model(self, use_test_set: bool = False) -> None:
        """
        Evaluate the model and display performance metrics.

        Args:
            use_test_set (bool): If True, evaluate on the test set instead of validation set.
        """
        generator = self.test_generator if use_test_set else self.val_generator
        if not self.model or not generator:
            raise ValueError("Model and generator must be initialized.")

        try:
            val_steps = generator.samples // generator.batch_size
            y_pred = self.model.predict(generator, steps=val_steps, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = generator.classes[:len(y_pred_classes)]

            class_labels = list(generator.class_indices.keys())
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred_classes, target_names=class_labels))

            cm = confusion_matrix(y_true, y_pred_classes)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_labels, yticklabels=class_labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()
            logger.info("Model evaluation completed on %s set.", "test" if use_test_set else "validation")
        except Exception as e:
            logger.error("Model evaluation failed: %s", e)
            raise

    def cross_validate(self) -> List[Dict]:
        """
        Perform k-fold cross-validation.

        Returns:
            List[Dict]: List of training histories for each fold.
        """
        try:
            kfold = KFold(n_splits=self.config['k_folds'], shuffle=True, random_state=42)
            histories = []
            fold_no = 1

            for train_idx, val_idx in kfold.split(np.zeros(self.train_generator.samples)):
                logger.info("Training fold %d/%d", fold_no, self.config['k_folds'])

                # Reset generators for this fold
                self.prepare_data(test_split=0)
                self.build_model()

                history = self.train_model(use_checkpoint=False)
                histories.append(history)
                fold_no += 1

            logger.info("Cross-validation completed with %d folds.", self.config['k_folds'])
            return histories
        except Exception as e:
            logger.error("Cross-validation failed: %s", e)
            raise

    def summarize_model(self) -> None:
        """Display the model architecture summary."""
        if not self.model:
            logger.warning("Model is not built yet.")
            return
        self.model.summary()
        logger.info("Model summary displayed.")

    def count_classes(self) -> Dict[str, int]:
        """
        Count the number of samples per class.

        Returns:
            Dict[str, int]: Dictionary mapping class labels to sample counts.
        """
        if not self.train_generator:
            logger.warning("Train generator is not initialized.")
            return {}

        try:
            class_counts = self.train_generator.classes
            unique, counts = np.unique(class_counts, return_counts=True)
            class_labels = list(self.train_generator.class_indices.keys())
            result = dict(zip(class_labels, counts))
            logger.info("Class counts: %s", result)
            return result
        except Exception as e:
            logger.error("Failed to count classes: %s", e)
            raise

    def plot_training_history(self) -> None:
        """Plot training and validation accuracy and loss."""
        if not self.history:
            logger.warning("No training history available. Train the model first.")
            return

        try:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Train Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
            plt.title('Accuracy over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'], label='Train Loss')
            plt.plot(self.history.history['val_loss'], label='Val Loss')
            plt.title('Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.show()
            logger.info("Training history plotted.")
        except Exception as e:
            logger.error("Failed to plot training history: %s", e)
            raise

    def list_available_classes(self) -> List[str]:
        """
        List available class labels.

        Returns:
            List[str]: List of class labels.
        """
        if not self.train_generator:
            logger.warning("Train generator is not initialized.")
            return []

        try:
            class_labels = list(self.train_generator.class_indices.keys())
            logger.info("Available classes: %s", class_labels)
            return class_labels
        except Exception as e:
            logger.error("Failed to list classes: %s", e)
            raise

    def is_model_trained(self) -> bool:
        """
        Check if the model is trained.

        Returns:
            bool: True if the model is trained, False otherwise.
        """
        result = self.model is not None and hasattr(self.model, 'history')
        logger.info("Model trained status: %s", result)
        return result 
