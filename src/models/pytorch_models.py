"""PyTorch models for Smart Home Activity Recognition with edge optimization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json

from ..utils.device_utils import get_device, get_model_size_mb, count_parameters
from ..utils.config import ModelConfig, EdgeConfig


class ActivityRecognitionCNN(nn.Module):
    """1D CNN model for smart home activity recognition."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=config.num_classes,  # Will be set based on input
            out_channels=config.conv_filters,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(config.conv_filters)
        
        self.conv2 = nn.Conv1d(
            in_channels=config.conv_filters,
            out_channels=config.conv_filters * 2,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(config.conv_filters * 2)
        
        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=config.pool_size)
        
        # Calculate flattened size
        self.flatten_size = self._calculate_flatten_size()
        
        # Dense layers
        self.fc1 = nn.Linear(self.flatten_size, config.dense_units)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(config.dense_units, config.num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_flatten_size(self) -> int:
        """Calculate the size after flattening conv layers."""
        # Assume input shape (batch, channels, sequence_length)
        # We'll use a dummy input to calculate this
        dummy_input = torch.randn(1, 3, 100)  # 3 sensors, 100 time steps
        x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return x.numel()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, num_sensors)
            
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Reshape input: (batch, sequence_length, num_sensors) -> (batch, num_sensors, sequence_length)
        x = x.transpose(1, 2)
        
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class EdgeOptimizedCNN(nn.Module):
    """Edge-optimized CNN with reduced parameters and quantization-friendly design."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the edge-optimized model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Use depthwise separable convolutions for efficiency
        self.conv1 = nn.Conv1d(
            in_channels=3,  # Fixed for 3 sensors
            out_channels=16,  # Reduced channels
            kernel_size=3,
            padding=1
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        
        # Global average pooling instead of dense layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Minimal dense layer
        self.classifier = nn.Linear(32, config.num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, num_sensors)
            
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Reshape input: (batch, sequence_length, num_sensors) -> (batch, num_sensors, sequence_length)
        x = x.transpose(1, 2)
        
        # First conv block
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        
        # Second conv block
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        
        return x


class ModelTrainer:
    """Trainer class for activity recognition models."""
    
    def __init__(
        self, 
        model: nn.Module, 
        config: ModelConfig,
        device: str = "cpu"
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            config: Model configuration
            device: Device to use for training
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Setup optimizer
        if config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=config.learning_rate
            )
        elif config.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=config.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = {
            "loss": [],
            "accuracy": []
        }
        self.val_history = {
            "loss": [],
            "accuracy": []
        }
    
    def train_epoch(
        self, 
        train_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self, 
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        save_checkpoint: bool = True,
        checkpoint_dir: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_checkpoint: Whether to save checkpoints
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        
        best_val_acc = 0.0
        
        for epoch in range(self.config.epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_history["loss"].append(train_loss)
            self.train_history["accuracy"].append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.val_history["loss"].append(val_loss)
                self.val_history["accuracy"].append(val_acc)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
                
                # Save best model
                if val_acc > best_val_acc and save_checkpoint:
                    best_val_acc = val_acc
                    self.save_checkpoint(checkpoint_dir, is_best=True)
            else:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
                )
        
        return {
            "train": self.train_history,
            "val": self.val_history
        }
    
    def save_checkpoint(
        self, 
        checkpoint_dir: str, 
        is_best: bool = False
    ) -> None:
        """Save model checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            is_best: Whether this is the best model
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "train_history": self.train_history,
            "val_history": self.val_history
        }
        
        filename = "best_model.pth" if is_best else "checkpoint.pth"
        torch.save(checkpoint, checkpoint_path / filename)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path / filename}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_history = checkpoint.get("train_history", {"loss": [], "accuracy": []})
        self.val_history = checkpoint.get("val_history", {"loss": [], "accuracy": []})
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


class ModelQuantizer:
    """Quantization utilities for edge deployment."""
    
    def __init__(self, model: nn.Module, config: EdgeConfig):
        """Initialize quantizer.
        
        Args:
            model: PyTorch model to quantize
            config: Edge configuration
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def quantize_model(self, calibration_data: torch.Tensor) -> nn.Module:
        """Quantize model using PyTorch quantization.
        
        Args:
            calibration_data: Data for calibration
            
        Returns:
            Quantized model
        """
        self.logger.info("Quantizing model...")
        
        # Set model to eval mode
        self.model.eval()
        
        # Prepare model for quantization
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model
        prepared_model = torch.quantization.prepare(self.model)
        
        # Calibrate with sample data
        with torch.no_grad():
            for i in range(min(len(calibration_data), self.config.calibration_samples)):
                sample = calibration_data[i:i+1]
                prepared_model(sample)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        self.logger.info("Model quantization completed")
        return quantized_model
    
    def get_model_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get model statistics.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with model statistics
        """
        stats = {
            "num_parameters": count_parameters(model),
            "model_size_mb": get_model_size_mb(model),
            "is_quantized": hasattr(model, 'qconfig') and model.qconfig is not None
        }
        
        return stats
