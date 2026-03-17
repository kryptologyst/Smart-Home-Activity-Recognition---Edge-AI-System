# Smart Home Activity Recognition - Edge AI System

Edge AI system for smart home activity recognition using multivariate time-series sensor data. This project demonstrates efficient deep learning models optimized for edge deployment with comprehensive evaluation and interactive demos.

## ⚠️ IMPORTANT DISCLAIMER

**NOT FOR SAFETY-CRITICAL USE** - This system is designed for research and educational purposes only. Do not rely on this system for any safety-critical applications, medical devices, or security systems.

## Project Overview

This system recognizes smart home activities (cooking, sleeping, walking) using synthetic multivariate time-series data from motion, door, temperature, and appliance sensors. It includes both TensorFlow and PyTorch implementations with edge optimization techniques.

### Key Features

- **Dual Framework Support**: TensorFlow and PyTorch implementations
- **Edge Optimization**: Quantization, pruning, and model compression
- **Multiple Deployment Targets**: TFLite, ONNX, CoreML, OpenVINO
- **Comprehensive Evaluation**: Accuracy, latency, memory, and energy metrics
- **Interactive Demo**: Streamlit app with real-time simulation
- **Device-Specific Configs**: Optimized for Raspberry Pi, Jetson Nano, Android, iOS
- **Reproducible Research**: Deterministic seeding and structured logging

## 📁 Project Structure

```
0774_Smart_Home_Activity_Recognition/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── tensorflow_models.py  # TensorFlow models
│   │   └── pytorch_models.py     # PyTorch models
│   ├── pipelines/               # Data processing
│   │   └── data_pipeline.py     # Data generation and processing
│   ├── export/                   # Model export and deployment
│   │   └── edge_deployment.py   # Edge runtime and profiling
│   └── utils/                    # Utilities
│       ├── config.py            # Configuration management
│       └── device_utils.py      # Device and performance utilities
├── data/                         # Data directories
│   ├── raw/                     # Raw data
│   └── processed/               # Processed data
├── configs/                      # Configuration files
├── models/                       # Trained models
├── checkpoints/                  # Training checkpoints
├── assets/                       # Results and visualizations
├── demo/                         # Interactive demo
│   └── app.py                   # Streamlit demo app
├── tests/                        # Unit tests
├── scripts/                      # Utility scripts
├── train.py                     # Main training script
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project configuration
└── README.md                    # This file
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, for faster training)
- 4GB+ RAM recommended

### Installation

1. **Clone and setup environment:**
   ```bash
   git clone https://github.com/kryptologyst/Smart-Home-Activity-Recognition---Edge-AI-System.git
   cd Smart-Home-Activity-Recognition---Edge-AI-System
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Install development dependencies (optional):**
   ```bash
   pip install -e ".[dev]"
   ```

### Basic Usage

1. **Train models:**
   ```bash
   # Train both TensorFlow and PyTorch models
   python train.py --framework both --model-type both
   
   # Train only TensorFlow edge-optimized model
   python train.py --framework tensorflow --model-type edge_optimized
   
   # Train with custom parameters
   python train.py --epochs 20 --batch-size 64 --device cuda
   ```

2. **Run interactive demo:**
   ```bash
   streamlit run demo/app.py
   ```

3. **View results:**
   - Training logs: Check console output
   - Model files: `models/` directory
   - Visualizations: `assets/` directory
   - Performance metrics: `assets/final_results.json`

## 🔧 Configuration

### Model Configuration

The system uses a hierarchical configuration system. Key parameters:

```yaml
data:
  num_samples_per_activity: 200
  sequence_length: 100
  num_sensors: 3
  activities: ["cooking", "sleeping", "walking"]

model:
  conv_filters: 32
  kernel_size: 3
  dense_units: 64
  epochs: 10
  batch_size: 32

edge:
  enable_quantization: true
  quantization_bits: 8
  enable_pruning: true
  pruning_ratio: 0.3
  max_latency_ms: 50.0
  max_model_size_mb: 5.0
```

### Device-Specific Optimization

The system automatically optimizes for different edge devices:

- **Raspberry Pi**: 100ms latency target, 10MB model size limit
- **Jetson Nano**: 50ms latency target, 20MB model size limit  
- **Android**: 30ms latency target, 5MB model size limit
- **iOS**: 25ms latency target, 5MB model size limit

## Model Architecture

### Standard CNN Model
- 1D Convolutional layers (32 → 64 filters)
- Batch normalization and dropout
- Dense layers (64 → 3 units)
- ~50K parameters, ~0.2MB

### Edge-Optimized Model
- Reduced convolutional layers (16 → 32 filters)
- Global average pooling instead of dense layers
- Minimal classifier
- ~15K parameters, ~0.06MB

## Performance Metrics

The system evaluates models across multiple dimensions:

### Accuracy Metrics
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
- Classification report

### Edge Performance
- **Latency**: Mean, P50, P95, P99 inference times
- **Throughput**: Frames per second (FPS)
- **Memory**: Peak RAM usage during inference
- **Model Size**: Compressed model size in MB
- **Energy**: Estimated energy consumption (when available)

### Robustness Testing
- Noise injection testing
- Sensor failure simulation
- Quantization impact analysis
- Cross-platform compatibility

## Model Export and Deployment

### Supported Formats

1. **TensorFlow Lite** (.tflite)
   - Quantized INT8 models
   - Optimized for mobile/embedded devices
   - Runtime: TFLite Runtime

2. **ONNX** (.onnx)
   - Cross-platform interoperability
   - Optimized inference graphs
   - Runtime: ONNX Runtime

3. **CoreML** (.mlmodel)
   - Apple device optimization
   - iOS/macOS deployment
   - Runtime: CoreML Framework

4. **OpenVINO** (.xml/.bin)
   - Intel hardware optimization
   - CPU/GPU/NPU acceleration
   - Runtime: OpenVINO Runtime

### Export Commands

```bash
# Export all formats
python scripts/export_models.py --input models/tensorflow_standard_model.h5

# Export specific format
python scripts/export_models.py --input models/pytorch_standard_model.pth --format onnx

# Benchmark exported models
python scripts/benchmark_models.py --models-dir models/
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_data_pipeline.py -v
```

## Interactive Demo

The Streamlit demo provides:

- **Real-time Simulation**: Generate sensor data for different activities
- **Model Comparison**: Switch between different models
- **Performance Monitoring**: Live latency and accuracy tracking
- **Visualization**: Interactive plots of sensor data and predictions
- **Edge Constraints**: Simulate real-world noise and sensor failures

### Demo Features

1. **Sensor Data Visualization**: Real-time plots of simulated sensor readings
2. **Activity Recognition**: Live inference with confidence scores
3. **Performance Dashboard**: Latency, throughput, and accuracy metrics
4. **Prediction History**: Track accuracy over time
5. **Model Information**: Display model architecture and parameters

## Research Applications

This project serves as a foundation for:

- **Edge AI Research**: Model compression and optimization techniques
- **IoT Analytics**: Time-series analysis for smart environments
- **Federated Learning**: Privacy-preserving distributed training
- **Hardware Acceleration**: GPU/NPU optimization strategies
- **Energy Efficiency**: Low-power inference algorithms

## Dataset Information

### Synthetic Data Generation

The system generates realistic sensor data using:

- **Temporal Patterns**: Sine wave variations for realistic sensor behavior
- **Activity-Specific Baselines**: Different sensor patterns for each activity
- **Noise Injection**: Gaussian noise for robustness testing
- **Sensor Failure Simulation**: Random sensor dropout scenarios

### Data Schema

```python
# Input shape: (batch_size, sequence_length, num_sensors)
# sequence_length: 100 time steps
# num_sensors: 3 (motion, appliance, door sensors)
# Activities: cooking, sleeping, walking
```

## 🛠️ Development

### Code Quality

The project follows modern Python practices:

- **Type Hints**: Full type annotation coverage
- **Documentation**: Google-style docstrings
- **Formatting**: Black code formatting, Ruff linting
- **Testing**: Pytest with comprehensive test coverage
- **CI/CD**: GitHub Actions for automated testing

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run linting and tests
5. Submit a pull request

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Requirements

### Core Dependencies
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- torch>=2.0.0
- tensorflow>=2.13.0
- streamlit>=1.25.0

### Edge Runtime Dependencies
- onnxruntime>=1.15.0
- tflite-runtime>=2.13.0
- openvino>=2023.0.0
- coremltools>=7.0

### Development Dependencies
- black>=23.0.0
- ruff>=0.0.280
- pytest>=7.4.0

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train.py --batch-size 16
   
   # Use CPU only
   python train.py --device cpu
   ```

2. **Model Loading Errors**
   ```bash
   # Ensure models are trained first
   python train.py --framework tensorflow
   
   # Check model files exist
   ls models/
   ```

3. **Streamlit Demo Issues**
   ```bash
   # Update Streamlit
   pip install --upgrade streamlit
   
   # Clear cache
   streamlit cache clear
   ```

### Performance Optimization

1. **For Training Speed**:
   - Use CUDA if available
   - Increase batch size
   - Reduce sequence length

2. **For Edge Deployment**:
   - Enable quantization
   - Use edge-optimized models
   - Reduce model complexity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow and PyTorch teams for excellent deep learning frameworks
- ONNX, TFLite, and OpenVINO communities for edge deployment tools
- Streamlit team for interactive web app framework
- Edge AI research community for optimization techniques

## Support

For questions and support:

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for general questions
- **Documentation**: Check the `docs/` directory for detailed guides

---

**Remember**: This system is for research and educational purposes only. **NOT FOR SAFETY-CRITICAL USE**.
# Smart-Home-Activity-Recognition---Edge-AI-System
