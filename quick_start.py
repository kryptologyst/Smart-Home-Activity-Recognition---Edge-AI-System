#!/usr/bin/env python3
"""
Smart Home Activity Recognition - Quick Start Script

This script provides a quick way to test the modernized system and compare
it with the original implementation.

NOT FOR SAFETY-CRITICAL USE - Research and educational purposes only.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def run_original_implementation():
    """Run the original implementation for comparison."""
    print("="*60)
    print("RUNNING ORIGINAL IMPLEMENTATION")
    print("="*60)
    
    try:
        import numpy as np
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # Original code from 0774.py
        def generate_data(activity, label, num_samples=200):
            base = {
                "cooking": [1.0, 0.3, 0.8],
                "sleeping": [0.1, 0.05, 0.0],
                "walking": [0.6, 0.2, 0.5]
            }[activity]
            data = np.random.normal(loc=base, scale=0.1, size=(num_samples, 100, 3))
            labels = np.full((num_samples,), label)
            return data, labels
        
        # Generate dataset
        cooking_X, cooking_y = generate_data("cooking", "cooking")
        sleeping_X, sleeping_y = generate_data("sleeping", "sleeping")
        walking_X, walking_y = generate_data("walking", "walking")
        
        # Combine all data
        X = np.vstack([cooking_X, sleeping_X, walking_X])
        y = np.concatenate([cooking_y, cooking_y, walking_y])
        
        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Build 1D CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(100, 3)),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Train the model
        start_time = time.time()
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        training_time = time.time() - start_time
        
        # Evaluate the model
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"✅ Original Implementation Results:")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Model Parameters: {model.count_params():,}")
        
        return acc, training_time, model.count_params()
        
    except Exception as e:
        print(f"❌ Original implementation failed: {e}")
        return None, None, None


def run_modernized_implementation():
    """Run the modernized implementation."""
    print("\n" + "="*60)
    print("RUNNING MODERNIZED IMPLEMENTATION")
    print("="*60)
    
    try:
        from utils.config import SystemConfig
        from pipelines.data_pipeline import SensorDataGenerator
        from models.tensorflow_models import TensorFlowActivityCNN, TensorFlowModelTrainer
        from utils.device_utils import set_deterministic_seed
        
        # Load configuration
        config = SystemConfig()
        
        # Set deterministic seed
        set_deterministic_seed(config.data.random_seed)
        
        # Generate data using modern pipeline
        generator = SensorDataGenerator(config.data)
        X_train, X_test, y_train, y_test = generator.generate_dataset()
        
        # Create modern model
        model = TensorFlowActivityCNN(config.model)
        
        # Initialize trainer
        trainer = TensorFlowModelTrainer(model, config.model, config.device)
        
        # Train model
        start_time = time.time()
        history = trainer.train(X_train, y_train, X_test, y_test, save_checkpoint=False)
        training_time = time.time() - start_time
        
        # Evaluate model
        loss, acc = trainer.evaluate(X_test, y_test)
        
        print(f"✅ Modernized Implementation Results:")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Model Parameters: {model.count_params():,}")
        print(f"   Features:")
        print(f"     - Deterministic seeding")
        print(f"     - Structured configuration")
        print(f"     - Comprehensive logging")
        print(f"     - Edge optimization ready")
        print(f"     - Multiple export formats")
        
        return acc, training_time, model.count_params()
        
    except Exception as e:
        print(f"❌ Modernized implementation failed: {e}")
        return None, None, None


def compare_implementations():
    """Compare original vs modernized implementations."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    # Run both implementations
    orig_acc, orig_time, orig_params = run_original_implementation()
    mod_acc, mod_time, mod_params = run_modernized_implementation()
    
    if orig_acc is not None and mod_acc is not None:
        print(f"\n📊 Performance Comparison:")
        print(f"   Accuracy:")
        print(f"     Original:  {orig_acc:.4f}")
        print(f"     Modernized: {mod_acc:.4f}")
        print(f"     Difference: {mod_acc - orig_acc:+.4f}")
        
        print(f"   Training Time:")
        print(f"     Original:  {orig_time:.2f}s")
        print(f"     Modernized: {mod_time:.2f}s")
        print(f"     Difference: {mod_time - orig_time:+.2f}s")
        
        print(f"   Model Parameters:")
        print(f"     Original:  {orig_params:,}")
        print(f"     Modernized: {mod_params:,}")
        print(f"     Difference: {mod_params - orig_params:+,}")
        
        print(f"\n🚀 Modernization Benefits:")
        print(f"   ✅ Clean, modular architecture")
        print(f"   ✅ Type hints and documentation")
        print(f"   ✅ Comprehensive configuration system")
        print(f"   ✅ Edge deployment capabilities")
        print(f"   ✅ Multiple framework support")
        print(f"   ✅ Performance profiling tools")
        print(f"   ✅ Interactive demo application")
        print(f"   ✅ CI/CD pipeline")
        print(f"   ✅ Safety disclaimers")
    
    print(f"\n⚠️  IMPORTANT: This system is NOT FOR SAFETY-CRITICAL USE")
    print(f"   Research and educational purposes only.")


def show_next_steps():
    """Show next steps for users."""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    print("1. 🏃‍♂️ Quick Start:")
    print("   python train.py --framework tensorflow --epochs 5")
    print("   streamlit run demo/app.py")
    
    print("\n2. 🔧 Advanced Usage:")
    print("   python train.py --framework both --model-type both")
    print("   python scripts/export_models.py")
    print("   python scripts/benchmark_models.py")
    
    print("\n3. 🧪 Testing:")
    print("   pytest tests/")
    print("   python -m pytest --cov=src")
    
    print("\n4. 📚 Documentation:")
    print("   - README.md: Complete setup guide")
    print("   - DISCLAIMER.md: Safety information")
    print("   - configs/: Configuration examples")
    
    print("\n5. 🎯 Edge Deployment:")
    print("   - Check configs/device_configs.yaml")
    print("   - Export models to TFLite/ONNX")
    print("   - Benchmark on target devices")
    
    print("\n6. 🔬 Research Extensions:")
    print("   - Add real sensor data")
    print("   - Implement federated learning")
    print("   - Add more activities")
    print("   - Optimize for specific hardware")


def main():
    """Main function."""
    print("🏠 Smart Home Activity Recognition - Modernization Demo")
    print("NOT FOR SAFETY-CRITICAL USE - Research and educational purposes only")
    print()
    
    try:
        compare_implementations()
        show_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check your installation and try again")
    
    print("\n" + "="*60)
    print("Thank you for using Smart Home Activity Recognition!")
    print("Remember: NOT FOR SAFETY-CRITICAL USE")
    print("="*60)


if __name__ == "__main__":
    main()
