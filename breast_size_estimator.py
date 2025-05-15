import cv2
import numpy as np
import tensorflow as tf
import math
import statistics
import keras_tuner as kt
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
                                     BatchNormalization, GlobalAveragePooling2D, Input)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

class BreastSizeEstimator:
    def __init__(self, image_size=(224, 224), batch_size=32, epochs=20, data_dir="dataset", use_transfer_learning=True):
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_dir = data_dir
        self.use_transfer_learning = use_transfer_learning
        self.model = None
        self.train_generator = None
        self.val_generator = None
        self.history = None

    def prepare_data(self):
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=25,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        self.train_generator = datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        self.val_generator = datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

    def build_model(self):
        if self.use_transfer_learning:
            base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(self.image_size[0], self.image_size[1], 3))
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.4)(x)
            predictions = Dense(self.train_generator.num_classes, activation='softmax')(x)

            self.model = Model(inputs=base_model.input, outputs=predictions)
        else:
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_size[0], self.image_size[1], 3)),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.4),
                Dense(self.train_generator.num_classes, activation='softmax')
            ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def hyperparameter_tuning(self):
        def model_builder(hp):
            hp_units = hp.Int('units', min_value=64, max_value=256, step=64)
            hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

            base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(self.image_size[0], self.image_size[1], 3))
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(hp_units, activation='relu')(x)
            x = Dropout(hp_dropout)(x)
            predictions = Dense(self.train_generator.num_classes, activation='softmax')(x)

            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model

        tuner = kt.Hyperband(
            model_builder,
            objective='val_accuracy',
            max_epochs=10,
            factor=3,
            directory='kt_tuning',
            project_name='breast_size_tuning'
        )

        stop_early = EarlyStopping(monitor='val_loss', patience=3)
        tuner.search(self.train_generator, validation_data=self.val_generator, epochs=10, callbacks=[stop_early])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.model = tuner.hypermodel.build(best_hps)

    def train_model(self):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

        self.history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=[early_stop, reduce_lr]
        )
        return self.history

    def fine_tune_model(self):
        if self.use_transfer_learning and hasattr(self.model, 'layers'):
            self.model.layers[0].trainable = True
            for layer in self.model.layers[0].layers[:100]:
                layer.trainable = False
            for layer in self.model.layers[0].layers[100:]:
                layer.trainable = True

            self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

            self.model.fit(
                self.train_generator,
                epochs=self.epochs // 2,
                validation_data=self.val_generator
            )

    def save_model(self, file_path="breast_size_classifier.h5"):
        self.model.save(file_path)

    def load_model(self, file_path="breast_size_classifier.h5"):
        self.model = load_model(file_path)

    def predict_breast_size(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.image_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = self.model.predict(img)
        class_index = np.argmax(predictions[0])
        class_labels = list(self.train_generator.class_indices.keys())
        return class_labels[class_index]

    def evaluate_model(self):
        val_steps = self.val_generator.samples // self.val_generator.batch_size
        y_pred = self.model.predict(self.val_generator, steps=val_steps)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = self.val_generator.classes[:len(y_pred_classes)]

        class_labels = list(self.val_generator.class_indices.keys())
        print(classification_report(y_true, y_pred_classes, target_names=class_labels))

        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def summarize_model(self):
        if self.model:
            self.model.summary()
        else:
            print("Model is not built yet.")

    def count_classes(self):
        if self.train_generator:
            class_counts = self.train_generator.classes
            unique, counts = np.unique(class_counts, return_counts=True)
            class_labels = list(self.train_generator.class_indices.keys())
            return dict(zip(class_labels, counts))
        return {}

    def plot_training_history(self):
        if self.history:
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
        else:
            print("No training history available. Train the model first.")

    def list_available_classes(self):
        if self.train_generator:
            return list(self.train_generator.class_indices.keys())
        return []

    def is_model_trained(self):
        return self.model is not None and hasattr(self.model, 'history')
