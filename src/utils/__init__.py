"""Utility functions for deterministic behavior and device management."""

import random
import numpy as np
import torch
import tensorflow as tf
from typing import Optional, Union
import logging
import os
from pathlib import Path


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seed for all random number generators.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # Environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def get_device(device_preference: Optional[str] = None) -> str:
    """Get the best available device for computation.
    
    Args:
        device_preference: Preferred device ('cpu', 'cuda', 'mps')
        
    Returns:
        Available device string
    """
    if device_preference == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif device_preference == "mps" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def create_directories(paths: Union[str, list]) -> None:
    """Create directories if they don't exist.
    
    Args:
        paths: Single path string or list of paths
    """
    if isinstance(paths, str):
        paths = [paths]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable format.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f}PB"


class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
        else:
            import time
            self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available() and hasattr(self.start_time, 'record'):
            self.end_time = torch.cuda.Event(enable_timing=True)
            self.end_time.record()
            torch.cuda.synchronize()
            elapsed_ms = self.start_time.elapsed_time(self.end_time)
            print(f"{self.name}: {elapsed_ms:.2f}ms")
        else:
            import time
            elapsed_s = time.time() - self.start_time
            print(f"{self.name}: {elapsed_s:.4f}s")


def validate_input_shape(input_shape: tuple, expected_shape: tuple) -> bool:
    """Validate input shape matches expected shape.
    
    Args:
        input_shape: Actual input shape
        expected_shape: Expected input shape
        
    Returns:
        True if shapes match, False otherwise
    """
    if len(input_shape) != len(expected_shape):
        return False
    
    for actual, expected in zip(input_shape, expected_shape):
        if expected != -1 and actual != expected:
            return False
    
    return True
