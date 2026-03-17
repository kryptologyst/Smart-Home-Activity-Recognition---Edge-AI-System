"""Configuration management for Smart Home Activity Recognition system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from omegaconf import OmegaConf
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data generation and processing."""
    
    # Dataset parameters
    num_samples_per_activity: int = 200
    sequence_length: int = 100
    num_sensors: int = 3
    test_size: float = 0.2
    random_seed: int = 42
    
    # Sensor simulation parameters
    noise_scale: float = 0.1
    activities: List[str] = field(default_factory=lambda: ["cooking", "sleeping", "walking"])
    
    # Sensor types and their baseline values
    sensor_baselines: Dict[str, List[float]] = field(default_factory=lambda: {
        "cooking": [1.0, 0.3, 0.8],
        "sleeping": [0.1, 0.05, 0.0],
        "walking": [0.6, 0.2, 0.5]
    })
    
    # Data paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    
    # Model architecture
    conv_filters: int = 32
    kernel_size: int = 3
    pool_size: int = 2
    dense_units: int = 64
    num_classes: int = 3
    
    # Training parameters
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
    # Model paths
    model_dir: str = "models"
    checkpoint_dir: str = "checkpoints"


@dataclass
class EdgeConfig:
    """Configuration for edge deployment and optimization."""
    
    # Target devices
    target_devices: List[str] = field(default_factory=lambda: ["cpu", "tflite", "onnx"])
    
    # Quantization settings
    enable_quantization: bool = True
    quantization_bits: int = 8
    calibration_samples: int = 100
    
    # Pruning settings
    enable_pruning: bool = True
    pruning_ratio: float = 0.3
    
    # Performance targets
    max_latency_ms: float = 50.0
    max_model_size_mb: float = 5.0
    max_memory_mb: float = 100.0


@dataclass
class IoTConfig:
    """Configuration for IoT communication and sensors."""
    
    # MQTT settings
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic_prefix: str = "smart_home/sensors"
    
    # Sensor simulation
    sampling_rate_hz: float = 1.0
    buffer_size: int = 1000
    
    # Communication settings
    enable_encryption: bool = False
    qos_level: int = 1


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation and metrics."""
    
    # Metrics to compute
    compute_accuracy: bool = True
    compute_latency: bool = True
    compute_memory: bool = True
    compute_energy: bool = False
    
    # Performance testing
    num_inference_runs: int = 1000
    warmup_runs: int = 100
    
    # Output settings
    save_predictions: bool = True
    generate_plots: bool = True
    output_dir: str = "assets"


@dataclass
class SystemConfig:
    """Main system configuration."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    iot: IoTConfig = field(default_factory=IoTConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # System settings
    device: str = "cpu"  # cpu, cuda, mps
    num_workers: int = 4
    log_level: str = "INFO"
    
    # Paths
    project_root: str = "."
    config_dir: str = "configs"
    assets_dir: str = "assets"


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        SystemConfig object with loaded configuration
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return OmegaConf.structured(SystemConfig(**config_dict))
    else:
        return SystemConfig()


def save_config(config: SystemConfig, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: SystemConfig object to save
        config_path: Path where to save the configuration
    """
    config_dict = OmegaConf.to_yaml(config)
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def get_device_config(device_type: str) -> Dict[str, Any]:
    """Get device-specific configuration.
    
    Args:
        device_type: Type of device (raspberry_pi, jetson, android, etc.)
        
    Returns:
        Dictionary with device-specific settings
    """
    device_configs = {
        "raspberry_pi": {
            "cpu_cores": 4,
            "memory_gb": 4,
            "max_freq_mhz": 1500,
            "supported_runtimes": ["tflite", "onnx"],
            "recommended_batch_size": 1,
            "power_consumption_w": 3.5
        },
        "jetson_nano": {
            "cpu_cores": 4,
            "memory_gb": 4,
            "gpu_cores": 128,
            "max_freq_mhz": 1400,
            "supported_runtimes": ["tflite", "onnx", "tensorrt"],
            "recommended_batch_size": 1,
            "power_consumption_w": 10.0
        },
        "android": {
            "cpu_cores": 8,
            "memory_gb": 6,
            "supported_runtimes": ["tflite", "onnx"],
            "recommended_batch_size": 1,
            "power_consumption_w": 5.0
        },
        "ios": {
            "cpu_cores": 6,
            "memory_gb": 4,
            "supported_runtimes": ["coreml", "onnx"],
            "recommended_batch_size": 1,
            "power_consumption_w": 4.0
        }
    }
    
    return device_configs.get(device_type, device_configs["raspberry_pi"])
