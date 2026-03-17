"""Model export and deployment utilities for edge devices."""

import torch
import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
import time
import psutil
import os

from ..utils.device_utils import get_device, PerformanceTimer
from ..utils.config import EdgeConfig, SystemConfig


class ModelExporter:
    """Export models to various edge deployment formats."""
    
    def __init__(self, config: EdgeConfig):
        """Initialize exporter.
        
        Args:
            config: Edge configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def export_to_onnx(
        self, 
        model: torch.nn.Module, 
        input_shape: Tuple[int, ...],
        output_path: str,
        opset_version: int = 11
    ) -> str:
        """Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            
        Returns:
            Path to exported ONNX model
        """
        self.logger.info("Exporting PyTorch model to ONNX...")
        
        model.eval()
        device = get_device()
        model = model.to(device)
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        self.logger.info(f"ONNX model exported to {output_path}")
        return output_path
    
    def export_to_tflite(
        self, 
        model: tf.keras.Model, 
        output_path: str,
        quantize: bool = True
    ) -> str:
        """Export TensorFlow model to TensorFlow Lite format.
        
        Args:
            model: TensorFlow model
            output_path: Path to save TFLite model
            quantize: Whether to quantize the model
            
        Returns:
            Path to exported TFLite model
        """
        self.logger.info("Exporting TensorFlow model to TFLite...")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        self.logger.info(f"TFLite model exported to {output_path}")
        return output_path
    
    def export_to_coreml(
        self, 
        model: torch.nn.Module, 
        input_shape: Tuple[int, ...],
        output_path: str
    ) -> str:
        """Export PyTorch model to CoreML format.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            output_path: Path to save CoreML model
            
        Returns:
            Path to exported CoreML model
        """
        try:
            import coremltools as ct
        except ImportError:
            self.logger.error("CoreML tools not installed. Install with: pip install coremltools")
            return ""
        
        self.logger.info("Exporting PyTorch model to CoreML...")
        
        model.eval()
        device = get_device()
        model = model.to(device)
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=dummy_input.shape)]
        )
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        coreml_model.save(output_path)
        
        self.logger.info(f"CoreML model exported to {output_path}")
        return output_path
    
    def export_to_openvino(
        self, 
        model_path: str, 
        output_dir: str
    ) -> str:
        """Export ONNX model to OpenVINO format.
        
        Args:
            model_path: Path to ONNX model
            output_dir: Directory to save OpenVINO model
            
        Returns:
            Path to exported OpenVINO model
        """
        try:
            from openvino.tools import mo
            from openvino.runtime import Core
        except ImportError:
            self.logger.error("OpenVINO not installed. Install with: pip install openvino")
            return ""
        
        self.logger.info("Exporting ONNX model to OpenVINO...")
        
        # Convert ONNX to OpenVINO IR
        mo.convert_model(
            input_model=model_path,
            output_dir=output_dir,
            compress_to_fp16=True
        )
        
        self.logger.info(f"OpenVINO model exported to {output_dir}")
        return output_dir


class EdgeRuntime:
    """Edge runtime for model inference."""
    
    def __init__(self, model_path: str, runtime_type: str = "onnx"):
        """Initialize edge runtime.
        
        Args:
            model_path: Path to model file
            runtime_type: Type of runtime ("onnx", "tflite", "openvino")
        """
        self.model_path = model_path
        self.runtime_type = runtime_type
        self.logger = logging.getLogger(__name__)
        self.session = None
        self._load_model()
    
    def _load_model(self):
        """Load model for inference."""
        if self.runtime_type == "onnx":
            self._load_onnx_model()
        elif self.runtime_type == "tflite":
            self._load_tflite_model()
        elif self.runtime_type == "openvino":
            self._load_openvino_model()
        else:
            raise ValueError(f"Unsupported runtime type: {self.runtime_type}")
    
    def _load_onnx_model(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            self.logger.error("ONNX Runtime not installed. Install with: pip install onnxruntime")
            return
        
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.logger.info("ONNX model loaded successfully")
    
    def _load_tflite_model(self):
        """Load TensorFlow Lite model."""
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            self.logger.error("TFLite Runtime not installed. Install with: pip install tflite-runtime")
            return
        
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.logger.info("TFLite model loaded successfully")
    
    def _load_openvino_model(self):
        """Load OpenVINO model."""
        try:
            from openvino.runtime import Core
        except ImportError:
            self.logger.error("OpenVINO not installed. Install with: pip install openvino")
            return
        
        core = Core()
        self.model = core.read_model(self.model_path)
        self.compiled_model = core.compile_model(self.model, "CPU")
        self.logger.info("OpenVINO model loaded successfully")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data.
        
        Args:
            input_data: Input data array
            
        Returns:
            Prediction results
        """
        if self.runtime_type == "onnx":
            return self._predict_onnx(input_data)
        elif self.runtime_type == "tflite":
            return self._predict_tflite(input_data)
        elif self.runtime_type == "openvino":
            return self._predict_openvino(input_data)
        else:
            raise ValueError(f"Unsupported runtime type: {self.runtime_type}")
    
    def _predict_onnx(self, input_data: np.ndarray) -> np.ndarray:
        """ONNX inference."""
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_data})
        return output[0]
    
    def _predict_tflite(self, input_data: np.ndarray) -> np.ndarray:
        """TFLite inference."""
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(output_details[0]['index'])
        return output
    
    def _predict_openvino(self, input_data: np.ndarray) -> np.ndarray:
        """OpenVINO inference."""
        output = self.compiled_model([input_data])
        return output[0]


class PerformanceProfiler:
    """Profile model performance on edge devices."""
    
    def __init__(self):
        """Initialize profiler."""
        self.logger = logging.getLogger(__name__)
    
    def profile_model(
        self, 
        runtime: EdgeRuntime, 
        input_data: np.ndarray,
        num_runs: int = 1000,
        warmup_runs: int = 100
    ) -> Dict[str, Any]:
        """Profile model performance.
        
        Args:
            runtime: Edge runtime instance
            input_data: Input data for profiling
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Profiling model performance with {num_runs} runs...")
        
        # Warmup runs
        for _ in range(warmup_runs):
            runtime.predict(input_data)
        
        # Performance runs
        latencies = []
        memory_usage = []
        
        for i in range(num_runs):
            # Measure memory before inference
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Measure latency
            start_time = time.time()
            runtime.predict(input_data)
            end_time = time.time()
            
            # Measure memory after inference
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
            memory_usage.append(memory_after - memory_before)
        
        # Calculate statistics
        latencies = np.array(latencies)
        memory_usage = np.array(memory_usage)
        
        metrics = {
            "latency_ms": {
                "mean": float(np.mean(latencies)),
                "std": float(np.std(latencies)),
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99))
            },
            "memory_mb": {
                "mean": float(np.mean(memory_usage)),
                "std": float(np.std(memory_usage)),
                "max": float(np.max(memory_usage))
            },
            "throughput_fps": float(1000 / np.mean(latencies)),
            "num_runs": num_runs
        }
        
        self.logger.info(f"Profiling completed. Mean latency: {metrics['latency_ms']['mean']:.2f}ms")
        
        return metrics
    
    def benchmark_models(
        self, 
        models: Dict[str, EdgeRuntime], 
        input_data: np.ndarray,
        num_runs: int = 1000
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark multiple models.
        
        Args:
            models: Dictionary of model names to runtime instances
            input_data: Input data for benchmarking
            num_runs: Number of inference runs
            
        Returns:
            Benchmark results for each model
        """
        results = {}
        
        for model_name, runtime in models.items():
            self.logger.info(f"Benchmarking {model_name}...")
            results[model_name] = self.profile_model(runtime, input_data, num_runs)
        
        return results


class DeviceConfigManager:
    """Manage device-specific configurations."""
    
    def __init__(self):
        """Initialize device config manager."""
        self.logger = logging.getLogger(__name__)
    
    def get_device_capabilities(self) -> Dict[str, Any]:
        """Get current device capabilities.
        
        Returns:
            Device capabilities dictionary
        """
        capabilities = {
            "cpu_count": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "platform": os.name,
            "python_version": os.sys.version,
            "torch_available": torch.cuda.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
        
        if torch.cuda.is_available():
            capabilities["cuda_device_count"] = torch.cuda.device_count()
            capabilities["cuda_device_name"] = torch.cuda.get_device_name(0)
        
        return capabilities
    
    def optimize_for_device(self, config: EdgeConfig, device_type: str) -> EdgeConfig:
        """Optimize configuration for specific device.
        
        Args:
            config: Edge configuration
            device_type: Target device type
            
        Returns:
            Optimized configuration
        """
        device_configs = {
            "raspberry_pi": {
                "max_latency_ms": 100.0,
                "max_model_size_mb": 10.0,
                "max_memory_mb": 200.0,
                "enable_quantization": True,
                "quantization_bits": 8
            },
            "jetson_nano": {
                "max_latency_ms": 50.0,
                "max_model_size_mb": 20.0,
                "max_memory_mb": 500.0,
                "enable_quantization": True,
                "quantization_bits": 8
            },
            "android": {
                "max_latency_ms": 30.0,
                "max_model_size_mb": 5.0,
                "max_memory_mb": 100.0,
                "enable_quantization": True,
                "quantization_bits": 8
            },
            "ios": {
                "max_latency_ms": 25.0,
                "max_model_size_mb": 5.0,
                "max_memory_mb": 100.0,
                "enable_quantization": True,
                "quantization_bits": 8
            }
        }
        
        if device_type in device_configs:
            device_config = device_configs[device_type]
            for key, value in device_config.items():
                setattr(config, key, value)
            
            self.logger.info(f"Configuration optimized for {device_type}")
        
        return config
