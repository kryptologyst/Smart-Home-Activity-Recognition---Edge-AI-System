"""TensorFlow models for Smart Home Activity Recognition."""

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import numpy as np

from ..utils.device_utils import set_deterministic_seed
from ..utils.config import ModelConfig, EdgeConfig


class TensorFlowActivityCNN(Model):
    """TensorFlow 1D CNN model for smart home activity recognition."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Convolutional layers
        self.conv1 = layers.Conv1D(
            filters=config.conv_filters,
            kernel_size=config.kernel_size,
            activation='relu',
            padding='same',
            input_shape=(config.sequence_length, config.num_sensors)
        )
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv1D(
            filters=config.conv_filters * 2,
            kernel_size=config.kernel_size,
            activation='relu',
            padding='same'
        )
        self.bn2 = layers.BatchNormalization()
        
        # Pooling
        self.pool = layers.MaxPooling1D(pool_size=config.pool_size)
        
        # Dense layers
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(config.dense_units, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.fc2 = layers.Dense(config.num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        """Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        
        return x


class EdgeOptimizedTensorFlowCNN(Model):
    """Edge-optimized TensorFlow CNN with reduced parameters."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the edge-optimized model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Reduced convolutional layers
        self.conv1 = layers.Conv1D(
            filters=16,  # Reduced from 32
            kernel_size=3,
            activation='relu',
            padding='same',
            input_shape=(config.sequence_length, config.num_sensors)
        )
        
        self.conv2 = layers.Conv1D(
            filters=32,  # Reduced from 64
            kernel_size=3,
            activation='relu',
            padding='same'
        )
        
        # Global average pooling instead of dense layers
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        
        # Minimal dense layer
        self.classifier = layers.Dense(config.num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        """Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        x = self.conv1(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = self.conv2(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        
        return x


class TensorFlowModelTrainer:
    """Trainer class for TensorFlow activity recognition models."""
    
    def __init__(
        self, 
        model: Model, 
        config: ModelConfig,
        device: str = "cpu"
    ):
        """Initialize trainer.
        
        Args:
            model: TensorFlow model
            config: Model configuration
            device: Device to use for training
        """
        self.model = model
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Setup optimizer
        if config.optimizer.lower() == "adam":
            self.optimizer = Adam(learning_rate=config.learning_rate)
        elif config.optimizer.lower() == "sgd":
            self.optimizer = SGD(learning_rate=config.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        
        # Compile model
        self.model.compile(
            optimizer=self.optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Training history
        self.history = None
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_checkpoint: bool = True,
        checkpoint_dir: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            save_checkpoint: Whether to save checkpoints
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        
        # Setup callbacks
        callbacks = []
        
        if save_checkpoint:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            checkpoint_callback = ModelCheckpoint(
                filepath=str(checkpoint_path / "best_model.h5"),
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy' if X_val is not None else 'accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_reduction = ReduceLROnPlateau(
            monitor='val_accuracy' if X_val is not None else 'accuracy',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reduction)
        
        # Train the model
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Tuple[float, float]:
        """Evaluate the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        return self.model.predict(X, verbose=0)
    
    def save_model(self, model_path: str) -> None:
        """Save the model.
        
        Args:
            model_path: Path to save the model
        """
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load the model.
        
        Args:
            model_path: Path to the model file
        """
        self.model = tf.keras.models.load_model(model_path)
        self.logger.info(f"Model loaded from {model_path}")


class TensorFlowQuantizer:
    """Quantization utilities for TensorFlow models."""
    
    def __init__(self, model: Model, config: EdgeConfig):
        """Initialize quantizer.
        
        Args:
            model: TensorFlow model to quantize
            config: Edge configuration
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def quantize_model(self, representative_dataset: np.ndarray) -> Model:
        """Quantize model using TensorFlow Lite quantization.
        
        Args:
            representative_dataset: Dataset for calibration
            
        Returns:
            Quantized TensorFlow Lite model
        """
        self.logger.info("Quantizing TensorFlow model...")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Set quantization parameters
        if self.config.quantization_bits == 8:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif self.config.quantization_bits == 16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        # Set representative dataset for calibration
        def representative_data_gen():
            for i in range(min(len(representative_dataset), self.config.calibration_samples)):
                yield [representative_dataset[i:i+1]]
        
        converter.representative_dataset = representative_data_gen
        
        # Convert
        quantized_model = converter.convert()
        
        self.logger.info("TensorFlow model quantization completed")
        return quantized_model
    
    def save_tflite_model(self, quantized_model: bytes, output_path: str) -> None:
        """Save quantized TensorFlow Lite model.
        
        Args:
            quantized_model: Quantized model bytes
            output_path: Path to save the model
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(quantized_model)
        
        self.logger.info(f"Quantized model saved to {output_path}")
    
    def get_model_info(self, model: Model) -> Dict[str, Any]:
        """Get model information.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with model information
        """
        # Count parameters
        total_params = model.count_params()
        
        # Get model size (approximate)
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        info = {
            "total_parameters": total_params,
            "model_size_mb": model_size_mb,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape
        }
        
        return info


def create_tensorflow_model(config: ModelConfig, model_type: str = "standard") -> Model:
    """Create a TensorFlow model based on configuration.
    
    Args:
        config: Model configuration
        model_type: Type of model ("standard" or "edge_optimized")
        
    Returns:
        TensorFlow model
    """
    if model_type == "standard":
        return TensorFlowActivityCNN(config)
    elif model_type == "edge_optimized":
        return EdgeOptimizedTensorFlowCNN(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_pytorch_model(config: ModelConfig, model_type: str = "standard") -> torch.nn.Module:
    """Create a PyTorch model based on configuration.
    
    Args:
        config: Model configuration
        model_type: Type of model ("standard" or "edge_optimized")
        
    Returns:
        PyTorch model
    """
    if model_type == "standard":
        return ActivityRecognitionCNN(config)
    elif model_type == "edge_optimized":
        return EdgeOptimizedCNN(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
