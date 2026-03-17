#!/usr/bin/env python3
"""
Smart Home Activity Recognition - Main Training Script

This script demonstrates the modernized Edge AI system for smart home activity recognition
using multivariate time-series sensor data. It includes both TensorFlow and PyTorch
implementations with edge optimization techniques.

NOT FOR SAFETY-CRITICAL USE - Research and educational purposes only.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from utils.device_utils import set_deterministic_seed, setup_logging, get_device
from utils.config import SystemConfig, load_config, save_config
from pipelines.data_pipeline import SensorDataGenerator, SensorDataProcessor, DataEvaluator
from models.tensorflow_models import (
    TensorFlowActivityCNN, 
    EdgeOptimizedTensorFlowCNN,
    TensorFlowModelTrainer,
    TensorFlowQuantizer
)
from models.pytorch_models import (
    ActivityRecognitionCNN,
    EdgeOptimizedCNN,
    ModelTrainer,
    ModelQuantizer
)
from export.edge_deployment import ModelExporter, PerformanceProfiler, DeviceConfigManager


def setup_environment(config: SystemConfig) -> None:
    """Setup the training environment.
    
    Args:
        config: System configuration
    """
    # Set deterministic seed
    set_deterministic_seed(config.data.random_seed)
    
    # Setup logging
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    directories = [
        config.data.raw_data_dir,
        config.data.processed_data_dir,
        config.model.model_dir,
        config.model.checkpoint_dir,
        config.evaluation.output_dir,
        "assets"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment setup completed")


def generate_and_prepare_data(config: SystemConfig) -> tuple:
    """Generate synthetic data and prepare it for training.
    
    Args:
        config: System configuration
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, data_generator)
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating synthetic smart home sensor data...")
    
    # Initialize data generator
    data_generator = SensorDataGenerator(config.data)
    
    # Generate dataset
    X_train, X_test, y_train, y_test = data_generator.generate_dataset()
    
    # Save dataset
    data_generator.save_dataset(
        X_train, X_test, y_train, y_test,
        config.data.processed_data_dir
    )
    
    logger.info(f"Dataset generated: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test, data_generator


def train_tensorflow_model(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray, 
    y_test: np.ndarray,
    config: SystemConfig,
    model_type: str = "standard"
) -> Dict[str, Any]:
    """Train TensorFlow model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: System configuration
        model_type: Type of model ("standard" or "edge_optimized")
        
    Returns:
        Training results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Training TensorFlow {model_type} model...")
    
    # Create model
    if model_type == "standard":
        model = TensorFlowActivityCNN(config.model)
    else:
        model = EdgeOptimizedTensorFlowCNN(config.model)
    
    # Initialize trainer
    trainer = TensorFlowModelTrainer(model, config.model, config.device)
    
    # Train model
    history = trainer.train(
        X_train, y_train,
        X_test, y_test,
        save_checkpoint=True,
        checkpoint_dir=config.model.checkpoint_dir
    )
    
    # Evaluate model
    loss, accuracy = trainer.evaluate(X_test, y_test)
    logger.info(f"TensorFlow {model_type} model accuracy: {accuracy:.4f}")
    
    # Save model
    model_path = f"{config.model.model_dir}/tensorflow_{model_type}_model.h5"
    trainer.save_model(model_path)
    
    # Get model info
    quantizer = TensorFlowQuantizer(model, config.edge)
    model_info = quantizer.get_model_info(model)
    
    results = {
        "model_type": f"tensorflow_{model_type}",
        "accuracy": accuracy,
        "loss": loss,
        "history": history,
        "model_info": model_info,
        "model_path": model_path
    }
    
    return results


def train_pytorch_model(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray, 
    y_test: np.ndarray,
    config: SystemConfig,
    model_type: str = "standard"
) -> Dict[str, Any]:
    """Train PyTorch model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: System configuration
        model_type: Type of model ("standard" or "edge_optimized")
        
    Returns:
        Training results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Training PyTorch {model_type} model...")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.model.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.model.batch_size, shuffle=False
    )
    
    # Create model
    if model_type == "standard":
        model = ActivityRecognitionCNN(config.model)
    else:
        model = EdgeOptimizedCNN(config.model)
    
    # Initialize trainer
    trainer = ModelTrainer(model, config.model, config.device)
    
    # Train model
    history = trainer.train(
        train_loader,
        test_loader,
        save_checkpoint=True,
        checkpoint_dir=config.model.checkpoint_dir
    )
    
    # Evaluate model
    val_loss, val_accuracy = trainer.validate(test_loader)
    logger.info(f"PyTorch {model_type} model accuracy: {val_accuracy:.2f}%")
    
    # Save model
    model_path = f"{config.model.model_dir}/pytorch_{model_type}_model.pth"
    torch.save(model.state_dict(), model_path)
    
    # Get model info
    quantizer = ModelQuantizer(model, config.edge)
    model_info = quantizer.get_model_stats(model)
    
    results = {
        "model_type": f"pytorch_{model_type}",
        "accuracy": val_accuracy / 100.0,  # Convert to 0-1 range
        "loss": val_loss,
        "history": history,
        "model_info": model_info,
        "model_path": model_path
    }
    
    return results


def export_and_benchmark_models(
    results: Dict[str, Dict[str, Any]],
    X_test: np.ndarray,
    config: SystemConfig
) -> Dict[str, Any]:
    """Export models to edge formats and benchmark performance.
    
    Args:
        results: Training results for all models
        X_test: Test data for benchmarking
        config: System configuration
        
    Returns:
        Benchmark results
    """
    logger = logging.getLogger(__name__)
    logger.info("Exporting models to edge formats and benchmarking...")
    
    exporter = ModelExporter(config.edge)
    profiler = PerformanceProfiler()
    device_manager = DeviceConfigManager()
    
    benchmark_results = {}
    
    # Get device capabilities
    device_capabilities = device_manager.get_device_capabilities()
    logger.info(f"Device capabilities: {device_capabilities}")
    
    # Export and benchmark each model
    for model_name, model_results in results.items():
        logger.info(f"Processing {model_name}...")
        
        model_benchmarks = {}
        
        # Export to different formats based on model type
        if "tensorflow" in model_name:
            # Load TensorFlow model
            model = tf.keras.models.load_model(model_results["model_path"])
            
            # Export to TFLite
            tflite_path = f"models/{model_name}_quantized.tflite"
            exporter.export_to_tflite(model, tflite_path, quantize=True)
            
            # Benchmark TFLite model
            try:
                from export.edge_deployment import EdgeRuntime
                tflite_runtime = EdgeRuntime(tflite_path, "tflite")
                tflite_metrics = profiler.profile_model(
                    tflite_runtime, X_test[:1], num_runs=1000
                )
                model_benchmarks["tflite"] = tflite_metrics
            except Exception as e:
                logger.warning(f"TFLite benchmarking failed: {e}")
        
        elif "pytorch" in model_name:
            # Load PyTorch model
            if "standard" in model_name:
                model = ActivityRecognitionCNN(config.model)
            else:
                model = EdgeOptimizedCNN(config.model)
            
            model.load_state_dict(torch.load(model_results["model_path"]))
            model.eval()
            
            # Export to ONNX
            onnx_path = f"models/{model_name}.onnx"
            input_shape = (config.data.sequence_length, config.data.num_sensors)
            exporter.export_to_onnx(model, input_shape, onnx_path)
            
            # Benchmark ONNX model
            try:
                from export.edge_deployment import EdgeRuntime
                onnx_runtime = EdgeRuntime(onnx_path, "onnx")
                onnx_metrics = profiler.profile_model(
                    onnx_runtime, X_test[:1], num_runs=1000
                )
                model_benchmarks["onnx"] = onnx_metrics
            except Exception as e:
                logger.warning(f"ONNX benchmarking failed: {e}")
        
        benchmark_results[model_name] = model_benchmarks
    
    return benchmark_results


def evaluate_and_visualize(
    results: Dict[str, Dict[str, Any]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    data_generator: SensorDataGenerator,
    config: SystemConfig
) -> None:
    """Evaluate models and create visualizations.
    
    Args:
        results: Training results for all models
        X_test: Test features
        y_test: Test labels
        data_generator: Data generator with label encoder
        config: System configuration
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating models and creating visualizations...")
    
    evaluator = DataEvaluator(data_generator.label_encoder)
    
    # Evaluate each model
    for model_name, model_results in results.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        if "tensorflow" in model_name:
            model = tf.keras.models.load_model(model_results["model_path"])
            predictions = model.predict(X_test)
        else:
            # Load PyTorch model
            if "standard" in model_name:
                model = ActivityRecognitionCNN(config.model)
            else:
                model = EdgeOptimizedCNN(config.model)
            
            model.load_state_dict(torch.load(model_results["model_path"]))
            model.eval()
            
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                predictions = model(X_test_tensor).numpy()
        
        # Evaluate predictions
        metrics = evaluator.evaluate_predictions(
            y_test, predictions,
            save_report=True,
            output_dir=f"{config.evaluation.output_dir}/{model_name}"
        )
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")
    
    # Create comparison plots
    create_comparison_plots(results, config.evaluation.output_dir)


def create_comparison_plots(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """Create comparison plots for all models.
    
    Args:
        results: Training results for all models
        output_dir: Output directory for plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    logger = logging.getLogger(__name__)
    logger.info("Creating comparison plots...")
    
    # Accuracy comparison
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model size comparison
    model_sizes = []
    for name in model_names:
        if "model_info" in results[name]:
            size_mb = results[name]["model_info"].get("model_size_mb", 0)
            model_sizes.append(size_mb)
        else:
            model_sizes.append(0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, model_sizes)
    plt.title("Model Size Comparison")
    plt.ylabel("Model Size (MB)")
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, size in zip(bars, model_sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{size:.2f}MB', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_size_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison plots saved to {output_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Smart Home Activity Recognition Training")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--framework", type=str, choices=["tensorflow", "pytorch", "both"], 
                      default="both", help="Framework to use")
    parser.add_argument("--model-type", type=str, choices=["standard", "edge_optimized", "both"],
                      default="both", help="Model type to train")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device:
        config.device = args.device
    if args.epochs:
        config.model.epochs = args.epochs
    if args.batch_size:
        config.model.batch_size = args.batch_size
    
    # Setup environment
    setup_environment(config)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Smart Home Activity Recognition training...")
    logger.info("NOT FOR SAFETY-CRITICAL USE - Research and educational purposes only")
    
    # Generate and prepare data
    X_train, X_test, y_train, y_test, data_generator = generate_and_prepare_data(config)
    
    # Train models
    results = {}
    
    frameworks = ["tensorflow", "pytorch"] if args.framework == "both" else [args.framework]
    model_types = ["standard", "edge_optimized"] if args.model_type == "both" else [args.model_type]
    
    for framework in frameworks:
        for model_type in model_types:
            model_name = f"{framework}_{model_type}"
            
            if framework == "tensorflow":
                results[model_name] = train_tensorflow_model(
                    X_train, y_train, X_test, y_test, config, model_type
                )
            else:
                results[model_name] = train_pytorch_model(
                    X_train, y_train, X_test, y_test, config, model_type
                )
    
    # Export and benchmark models
    benchmark_results = export_and_benchmark_models(results, X_test, config)
    
    # Evaluate and visualize
    evaluate_and_visualize(results, X_test, y_test, data_generator, config)
    
    # Save final results
    final_results = {
        "training_results": results,
        "benchmark_results": benchmark_results,
        "config": config.__dict__
    }
    
    import json
    with open("assets/final_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logger.info("Training completed successfully!")
    logger.info("Results saved to assets/final_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("SMART HOME ACTIVITY RECOGNITION - TRAINING SUMMARY")
    print("="*60)
    print("NOT FOR SAFETY-CRITICAL USE - Research and educational purposes only")
    print()
    
    for model_name, model_results in results.items():
        print(f"{model_name.upper()}:")
        print(f"  Accuracy: {model_results['accuracy']:.4f}")
        if "model_info" in model_results:
            info = model_results["model_info"]
            if "model_size_mb" in info:
                print(f"  Model Size: {info['model_size_mb']:.2f} MB")
            if "total_parameters" in info:
                print(f"  Parameters: {info['total_parameters']:,}")
        print()
    
    print("Check assets/ directory for detailed results and visualizations.")
    print("="*60)


if __name__ == "__main__":
    main()
