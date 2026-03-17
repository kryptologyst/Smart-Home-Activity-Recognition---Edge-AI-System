"""Data pipeline for Smart Home Activity Recognition."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import logging
from pathlib import Path
import pickle
import json

from ..utils.device_utils import set_deterministic_seed
from ..utils.config import DataConfig


class SensorDataGenerator:
    """Generate synthetic smart home sensor data for activity recognition."""
    
    def __init__(self, config: DataConfig):
        """Initialize data generator with configuration.
        
        Args:
            config: Data configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set deterministic seed
        set_deterministic_seed(config.random_seed)
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(config.activities)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
    def generate_activity_data(
        self, 
        activity: str, 
        num_samples: int,
        add_noise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sensor data for a specific activity.
        
        Args:
            activity: Activity name
            num_samples: Number of samples to generate
            add_noise: Whether to add noise to the data
            
        Returns:
            Tuple of (sensor_data, labels)
        """
        if activity not in self.config.activities:
            raise ValueError(f"Unknown activity: {activity}")
        
        # Get baseline sensor values for this activity
        baseline = np.array(self.config.sensor_baselines[activity])
        
        # Generate time series data
        data = np.zeros((num_samples, self.config.sequence_length, self.config.num_sensors))
        labels = np.full(num_samples, activity)
        
        for i in range(num_samples):
            # Generate time series with some temporal correlation
            for t in range(self.config.sequence_length):
                # Add some temporal variation
                temporal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * t / self.config.sequence_length)
                
                if add_noise:
                    noise = np.random.normal(0, self.config.noise_scale, self.config.num_sensors)
                    data[i, t, :] = baseline * temporal_factor + noise
                else:
                    data[i, t, :] = baseline * temporal_factor
        
        return data, labels
    
    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate complete dataset with train/test split.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Generating synthetic smart home sensor dataset...")
        
        all_data = []
        all_labels = []
        
        # Generate data for each activity
        for activity in self.config.activities:
            self.logger.info(f"Generating {self.config.num_samples_per_activity} samples for '{activity}'")
            data, labels = self.generate_activity_data(
                activity, 
                self.config.num_samples_per_activity
            )
            all_data.append(data)
            all_labels.append(labels)
        
        # Combine all data
        X = np.vstack(all_data)
        y = np.concatenate(all_labels)
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.config.test_size, 
            random_state=self.config.random_seed,
            stratify=y_encoded
        )
        
        self.logger.info(f"Dataset generated: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_dataset(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray,
        save_dir: str
    ) -> None:
        """Save dataset to disk.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            save_dir: Directory to save the dataset
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save data arrays
        np.save(save_path / "X_train.npy", X_train)
        np.save(save_path / "X_test.npy", X_test)
        np.save(save_path / "y_train.npy", y_train)
        np.save(save_path / "y_test.npy", y_test)
        
        # Save metadata
        metadata = {
            "num_train_samples": len(X_train),
            "num_test_samples": len(X_test),
            "sequence_length": self.config.sequence_length,
            "num_sensors": self.config.num_sensors,
            "activities": self.config.activities,
            "class_mapping": dict(zip(
                range(len(self.label_encoder.classes_)), 
                self.label_encoder.classes_
            ))
        }
        
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save label encoder
        with open(save_path / "label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        self.logger.info(f"Dataset saved to {save_path}")
    
    def load_dataset(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset from disk.
        
        Args:
            data_dir: Directory containing the dataset
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        data_path = Path(data_dir)
        
        # Load data arrays
        X_train = np.load(data_path / "X_train.npy")
        X_test = np.load(data_path / "X_test.npy")
        y_train = np.load(data_path / "y_train.npy")
        y_test = np.load(data_path / "y_test.npy")
        
        # Load label encoder
        with open(data_path / "label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.logger.info(f"Dataset loaded from {data_path}")
        
        return X_train, X_test, y_train, y_test


class SensorDataProcessor:
    """Process and preprocess sensor data for edge deployment."""
    
    def __init__(self, config: DataConfig):
        """Initialize data processor.
        
        Args:
            config: Data configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_scaler(self, X: np.ndarray) -> None:
        """Fit the scaler on training data.
        
        Args:
            X: Training data
        """
        # Reshape data for scaling (samples * time_steps, features)
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_reshaped)
        self.is_fitted = True
        self.logger.info("Scaler fitted on training data")
    
    def transform_data(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming data")
        
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled
    
    def normalize_data(self, X: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range.
        
        Args:
            X: Input data
            
        Returns:
            Normalized data
        """
        X_min = X.min(axis=(0, 1), keepdims=True)
        X_max = X.max(axis=(0, 1), keepdims=True)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        return X_norm
    
    def add_noise(self, X: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to data for robustness testing.
        
        Args:
            X: Input data
            noise_level: Standard deviation of noise
            
        Returns:
            Noisy data
        """
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise
    
    def simulate_sensor_failure(self, X: np.ndarray, failure_rate: float = 0.1) -> np.ndarray:
        """Simulate sensor failure by zeroing out random sensors.
        
        Args:
            X: Input data
            failure_rate: Probability of sensor failure
            
        Returns:
            Data with simulated sensor failures
        """
        X_failed = X.copy()
        mask = np.random.random(X.shape) < failure_rate
        X_failed[mask] = 0
        return X_failed


class DataEvaluator:
    """Evaluate data quality and model performance."""
    
    def __init__(self, label_encoder: LabelEncoder):
        """Initialize evaluator.
        
        Args:
            label_encoder: Fitted label encoder
        """
        self.label_encoder = label_encoder
        self.logger = logging.getLogger(__name__)
    
    def evaluate_predictions(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_report: bool = True,
        output_dir: str = "assets"
    ) -> Dict[str, Any]:
        """Evaluate model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_report: Whether to save detailed report
            output_dir: Directory to save reports
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert predictions to class labels
        if y_pred.ndim > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = y_pred
        
        # Calculate metrics
        accuracy = np.mean(y_pred_classes == y_true)
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_true, y_pred_classes, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        metrics = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names.tolist()
        }
        
        if save_report:
            self._save_evaluation_report(metrics, output_dir)
        
        return metrics
    
    def _save_evaluation_report(self, metrics: Dict[str, Any], output_dir: str) -> None:
        """Save evaluation report to disk.
        
        Args:
            metrics: Evaluation metrics
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        with open(output_path / "evaluation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Evaluation report saved to {output_path}")
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray, 
        class_names: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
