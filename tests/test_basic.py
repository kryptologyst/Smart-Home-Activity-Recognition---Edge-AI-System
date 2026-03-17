"""Tests for Smart Home Activity Recognition system."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import SystemConfig, DataConfig, ModelConfig
from pipelines.data_pipeline import SensorDataGenerator
from utils.device_utils import set_deterministic_seed


class TestConfiguration:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = SystemConfig()
        assert config.data.num_samples_per_activity == 200
        assert config.model.epochs == 10
        assert config.edge.enable_quantization is True
    
    def test_data_config(self):
        """Test data configuration."""
        data_config = DataConfig()
        assert data_config.sequence_length == 100
        assert data_config.num_sensors == 3
        assert len(data_config.activities) == 3
    
    def test_model_config(self):
        """Test model configuration."""
        model_config = ModelConfig()
        assert model_config.conv_filters == 32
        assert model_config.num_classes == 3
        assert model_config.optimizer == "adam"


class TestDataPipeline:
    """Test data pipeline functionality."""
    
    def test_sensor_data_generator(self):
        """Test sensor data generation."""
        config = DataConfig()
        generator = SensorDataGenerator(config)
        
        # Test data generation for each activity
        for activity in config.activities:
            data, labels = generator.generate_activity_data(activity, 10)
            assert data.shape == (10, config.sequence_length, config.num_sensors)
            assert len(labels) == 10
            assert all(label == activity for label in labels)
    
    def test_dataset_generation(self):
        """Test complete dataset generation."""
        config = DataConfig()
        generator = SensorDataGenerator(config)
        
        X_train, X_test, y_train, y_test = generator.generate_dataset()
        
        # Check shapes
        assert X_train.shape[0] == int(config.num_samples_per_activity * len(config.activities) * (1 - config.test_size))
        assert X_test.shape[0] == int(config.num_samples_per_activity * len(config.activities) * config.test_size)
        assert X_train.shape[1:] == (config.sequence_length, config.num_sensors)
        assert X_test.shape[1:] == (config.sequence_length, config.num_sensors)
        
        # Check labels
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        assert all(0 <= label < len(config.activities) for label in y_train)
        assert all(0 <= label < len(config.activities) for label in y_test)
    
    def test_deterministic_seeding(self):
        """Test that seeding produces reproducible results."""
        config = DataConfig()
        config.random_seed = 42
        
        # Generate dataset twice with same seed
        generator1 = SensorDataGenerator(config)
        X_train1, X_test1, y_train1, y_test1 = generator1.generate_dataset()
        
        generator2 = SensorDataGenerator(config)
        X_train2, X_test2, y_train2, y_test2 = generator2.generate_dataset()
        
        # Results should be identical
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)


class TestDeviceUtils:
    """Test device utilities."""
    
    def test_deterministic_seed(self):
        """Test deterministic seeding."""
        # This test ensures seeding doesn't raise exceptions
        set_deterministic_seed(42)
        assert True  # If we get here, no exception was raised
    
    def test_device_detection(self):
        """Test device detection."""
        from utils.device_utils import get_device
        
        device = get_device()
        assert device in ["cpu", "cuda", "mps"]
        
        # Test with specific preference
        device_cpu = get_device("cpu")
        assert device_cpu == "cpu"


class TestModelArchitecture:
    """Test model architecture."""
    
    def test_tensorflow_model_creation(self):
        """Test TensorFlow model creation."""
        try:
            from models.tensorflow_models import TensorFlowActivityCNN, EdgeOptimizedTensorFlowCNN
            
            config = ModelConfig()
            
            # Test standard model
            model_standard = TensorFlowActivityCNN(config)
            assert model_standard is not None
            
            # Test edge-optimized model
            model_edge = EdgeOptimizedTensorFlowCNN(config)
            assert model_edge is not None
            
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_pytorch_model_creation(self):
        """Test PyTorch model creation."""
        try:
            import torch
            from models.pytorch_models import ActivityRecognitionCNN, EdgeOptimizedCNN
            
            config = ModelConfig()
            
            # Test standard model
            model_standard = ActivityRecognitionCNN(config)
            assert model_standard is not None
            
            # Test edge-optimized model
            model_edge = EdgeOptimizedCNN(config)
            assert model_edge is not None
            
        except ImportError:
            pytest.skip("PyTorch not available")


if __name__ == "__main__":
    pytest.main([__file__])
