#!/usr/bin/env python3
"""
Advanced Brain Tumor Detection System
Author: Krish Narula
Created: 2025

A comprehensive deep learning solution for automated brain tumor detection 
using transfer learning with VGG16 architecture and advanced image processing.

This system provides:
- Automated brain MRI analysis
- High-accuracy tumor classification
- Comprehensive model evaluation
- Professional reporting and visualization
- Command-line interface for easy deployment
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
import cv2
from imutils import paths

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    Input, Dense, AveragePooling2D, Dropout, 
    Flatten, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, 
    ModelCheckpoint, TensorBoard
)


@dataclass
class Config:
    """Configuration class for the Brain Tumor Detection System"""
    # Data parameters
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 16
    test_size: float = 0.15
    validation_size: float = 0.15
    random_state: int = 42
    
    # Model parameters
    learning_rate: float = 1e-4
    epochs: int = 25
    patience: int = 7
    
    # Paths
    dataset_path: str = "./dataset/brain_tumor_dataset"
    model_save_path: str = "./models"
    results_path: str = "./results"
    logs_path: str = "./logs"
    
    # Model architecture
    dense_units: int = 128
    dropout_rate: float = 0.5
    use_batch_norm: bool = True
    
    def __post_init__(self):
        """Create necessary directories"""
        for path in [self.model_save_path, self.results_path, self.logs_path]:
            Path(path).mkdir(parents=True, exist_ok=True)


class Logger:
    """Professional logging system for the project"""
    
    def __init__(self, name: str = "BrainTumorDetector", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler
            file_handler = logging.FileHandler(
                f'logs/brain_tumor_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, msg: str) -> None:
        self.logger.info(msg)
    
    def error(self, msg: str) -> None:
        self.logger.error(msg)
    
    def warning(self, msg: str) -> None:
        self.logger.warning(msg)


class DataProcessor:
    """Advanced data processing and augmentation system"""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.label_binarizer = LabelBinarizer()
        
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load and preprocess brain MRI images with advanced augmentation
        
        Returns:
            Tuple of (images, labels, class_names)
        """
        self.logger.info("Starting data loading and preprocessing...")
        
        if not os.path.exists(self.config.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.config.dataset_path}")
        
        image_paths = list(paths.list_images(self.config.dataset_path))
        self.logger.info(f"Found {len(image_paths)} images in dataset")
        
        images = []
        labels = []
        
        for idx, image_path in enumerate(image_paths):
            try:
                # Extract label from directory structure
                label = image_path.split(os.path.sep)[-2]
                
                # Load and preprocess image
                image = cv2.imread(image_path)
                if image is None:
                    self.logger.warning(f"Could not load image: {image_path}")
                    continue
                
                # Resize and normalize
                image = cv2.resize(image, self.config.image_size)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                
                # Apply histogram equalization for better contrast
                image = self._enhance_image_quality(image)
                
                images.append(image)
                labels.append(label)
                
                if (idx + 1) % 50 == 0:
                    self.logger.info(f"Processed {idx + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {str(e)}")
                continue
        
        # Convert to numpy arrays and normalize
        images = np.array(images, dtype=np.float32) / 255.0
        labels = np.array(labels)
        
        # Encode labels
        labels_encoded = self.label_binarizer.fit_transform(labels)
        labels_categorical = to_categorical(labels_encoded)
        
        class_names = self.label_binarizer.classes_.tolist()
        
        self.logger.info(f"Data preprocessing complete:")
        self.logger.info(f"- Total samples: {len(images)}")
        self.logger.info(f"- Image shape: {images[0].shape}")
        self.logger.info(f"- Classes: {class_names}")
        self.logger.info(f"- Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return images, labels_categorical, class_names
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced image enhancement techniques"""
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Apply Gaussian blur for noise reduction
        enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
        
        return enhanced
    
    def create_data_generators(self) -> ImageDataGenerator:
        """Create advanced data augmentation generators"""
        train_augmentation = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            preprocessing_function=self._custom_preprocessing
        )
        
        return train_augmentation
    
    def _custom_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Custom preprocessing function for data augmentation"""
        # Add slight random noise
        noise = np.random.normal(0, 0.01, image.shape)
        image = np.clip(image + noise, 0, 1)
        return image


class BrainTumorClassifier:
    """
    Advanced Brain Tumor Detection System by Krish Narula
    
    A state-of-the-art deep learning system for automated brain tumor detection
    using transfer learning with VGG16 and advanced computer vision techniques.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = Logger("KrishNarula_BrainTumorClassifier")
        self.model = None
        self.history = None
        self.class_names = None
        self.data_processor = DataProcessor(self.config, self.logger)
        
        # Set up TensorFlow
        self._setup_tensorflow()
        
        self.logger.info("=" * 60)
        self.logger.info("Brain Tumor Detection System Initialized")
        self.logger.info("Author: Krish Narula")
        self.logger.info("=" * 60)
    
    def _setup_tensorflow(self):
        """Configure TensorFlow for optimal performance"""
        # Set memory growth for GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"GPU acceleration enabled: {len(gpus)} GPU(s) found")
            except RuntimeError as e:
                self.logger.warning(f"GPU setup failed: {e}")
        else:
            self.logger.info("Running on CPU - consider using GPU for faster training")
    
    def build_model(self, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Model:
        """
        Build advanced transfer learning model with VGG16 backbone
        
        Args:
            input_shape: Input image dimensions
            
        Returns:
            Compiled Keras model
        """
        self.logger.info("Building advanced neural network architecture...")
        
        # Load pre-trained VGG16 without top layers
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_tensor=Input(shape=input_shape),
            input_shape=input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        if self.config.use_batch_norm:
            x = BatchNormalization(name='batch_norm_1')(x)
        
        x = Dense(
            self.config.dense_units, 
            activation='relu', 
            name='dense_1',
            kernel_initializer='he_normal'
        )(x)
        
        x = Dropout(self.config.dropout_rate, name='dropout_1')(x)
        
        if self.config.use_batch_norm:
            x = BatchNormalization(name='batch_norm_2')(x)
        
        x = Dense(
            self.config.dense_units // 2, 
            activation='relu', 
            name='dense_2',
            kernel_initializer='he_normal'
        )(x)
        
        x = Dropout(self.config.dropout_rate / 2, name='dropout_2')(x)
        
        # Output layer
        predictions = Dense(
            2, 
            activation='softmax', 
            name='predictions',
            kernel_initializer='glorot_uniform'
        )(x)
        
        # Create final model
        model = Model(inputs=base_model.input, outputs=predictions, name='KrishNarula_BrainTumorDetector')
        
        # Compile with advanced optimizer
        optimizer = Adam(
            learning_rate=self.config.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        
        # Log model summary
        self.logger.info("Model architecture created successfully:")
        self.logger.info(f"- Total parameters: {model.count_params():,}")
        self.logger.info(f"- Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        return model
    
    def train(self, save_model: bool = True) -> Dict[str, Any]:
        """
        Train the brain tumor detection model with advanced techniques
        
        Args:
            save_model: Whether to save the trained model
            
        Returns:
            Training history and metrics
        """
        self.logger.info("Starting model training...")
        start_time = datetime.now()
        
        # Load and preprocess data
        images, labels, class_names = self.data_processor.load_and_preprocess_data()
        self.class_names = class_names
        
        # Split data with stratification
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config.validation_size / (1 - self.config.test_size),
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Create data generators
        train_generator = self.data_processor.create_data_generators()
        
        # Calculate steps
        train_steps = len(X_train) // self.config.batch_size
        val_steps = len(X_val) // self.config.batch_size
        
        # Setup callbacks
        callbacks = self._get_callbacks()
        
        # Train the model
        self.logger.info("Training neural network...")
        
        history = self.model.fit(
            train_generator.flow(X_train, y_train, batch_size=self.config.batch_size),
            steps_per_epoch=train_steps,
            epochs=self.config.epochs,
            validation_data=(X_val, y_val),
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        
        # Evaluate on test set
        self.logger.info("Evaluating model on test set...")
        test_results = self.evaluate(X_test, y_test)
        
        # Training summary
        training_time = datetime.now() - start_time
        self.logger.info(f"Training completed in {training_time}")
        self.logger.info(f"Final test accuracy: {test_results['accuracy']:.4f}")
        
        # Save model
        if save_model:
            model_path = os.path.join(self.config.model_save_path, 'krish_narula_brain_tumor_model.h5')
            self.model.save(model_path)
            self.logger.info(f"Model saved to {model_path}")
        
        # Generate comprehensive report
        report = self._generate_training_report(history, test_results, training_time)
        
        return report
    
    def _get_callbacks(self) -> List:
        """Create advanced training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.config.model_save_path, 'best_model_checkpoint.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(self.config.logs_path, f'tensorboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive model evaluation with advanced metrics"""
        predictions = self.model.predict(X_test, batch_size=self.config.batch_size)
        y_pred_classes = np.argmax(predictions, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate comprehensive metrics
        metrics = {
            'accuracy': accuracy_score(y_true_classes, y_pred_classes),
            'precision': precision_score(y_true_classes, y_pred_classes, average='weighted'),
            'recall': recall_score(y_true_classes, y_pred_classes, average='weighted'),
            'f1_score': f1_score(y_true_classes, y_pred_classes, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true_classes, y_pred_classes),
            'classification_report': classification_report(
                y_true_classes, y_pred_classes, 
                target_names=self.class_names, 
                output_dict=True
            )
        }
        
        return metrics
    
    def _generate_training_report(self, history, test_results: Dict, training_time) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            'author': 'Krish Narula',
            'timestamp': datetime.now().isoformat(),
            'training_time': str(training_time),
            'model_config': {
                'architecture': 'VGG16 Transfer Learning',
                'input_shape': self.config.image_size + (3,),
                'batch_size': self.config.batch_size,
                'epochs': self.config.epochs,
                'learning_rate': self.config.learning_rate
            },
            'performance': test_results,
            'training_history': {
                'final_train_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1])
            }
        }
        
        # Save report
        report_path = os.path.join(self.config.results_path, f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Training report saved to {report_path}")
        
        return report
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations and save them"""
        if self.history is None:
            self.logger.warning("No training history available for visualization")
            return
        
        self.logger.info("Creating comprehensive visualizations...")
        
        # Set style
        plt.style.use('default')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Brain Tumor Detection Model Analysis by Krish Narula', fontsize=16, y=0.98)
        
        # Training history plots
        epochs = range(1, len(self.history.history['accuracy']) + 1)
        
        # Accuracy plot
        axes[0, 0].plot(epochs, self.history.history['accuracy'], 'bo-', label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(epochs, self.history.history['val_accuracy'], 'ro-', label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(epochs, self.history.history['loss'], 'bo-', label='Training Loss', linewidth=2)
        axes[0, 1].plot(epochs, self.history.history['val_loss'], 'ro-', label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision plot
        if 'precision' in self.history.history:
            axes[1, 0].plot(epochs, self.history.history['precision'], 'go-', label='Training Precision', linewidth=2)
            axes[1, 0].plot(epochs, self.history.history['val_precision'], 'mo-', label='Validation Precision', linewidth=2)
            axes[1, 0].set_title('Model Precision Over Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall plot
        if 'recall' in self.history.history:
            axes[1, 1].plot(epochs, self.history.history['recall'], 'co-', label='Training Recall', linewidth=2)
            axes[1, 1].plot(epochs, self.history.history['val_recall'], 'yo-', label='Validation Recall', linewidth=2)
            axes[1, 1].set_title('Model Recall Over Time')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        viz_path = os.path.join(self.config.results_path, f'training_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Visualizations saved to {viz_path}")


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(
        description="Advanced Brain Tumor Detection System by Krish Narula",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python brain_tumor_classifier.py --train --epochs 30
  python brain_tumor_classifier.py --train --config custom_config.json
  python brain_tumor_classifier.py --evaluate --model saved_model.h5
        """
    )
    
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate existing model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Path to saved model')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Override with command line arguments
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # Initialize classifier
    classifier = BrainTumorClassifier(config)
    
    if args.train:
        print("\n" + "="*80)
        print("KRISH NARULA'S BRAIN TUMOR DETECTION SYSTEM")
        print("Advanced Deep Learning for Medical Image Analysis")
        print("="*80)
        
        # Train the model
        report = classifier.train()
        
        # Create visualizations
        classifier.create_visualizations()
        
        print(f"\nTraining completed successfully!")
        print(f"Final accuracy: {report['performance']['accuracy']:.4f}")
        print(f"Model saved and ready for deployment.")
        
    elif args.evaluate:
        if not args.model or not os.path.exists(args.model):
            print("Error: Please provide a valid model path for evaluation")
            return
        
        # Load and evaluate model
        classifier.model = tf.keras.models.load_model(args.model)
        print(f"Model loaded from {args.model}")
        print("Evaluation functionality would be implemented here")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 