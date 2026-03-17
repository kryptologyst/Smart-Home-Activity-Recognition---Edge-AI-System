"""
Smart Home Activity Recognition - Interactive Demo

This Streamlit app provides an interactive demonstration of the smart home
activity recognition system, simulating edge device constraints and real-time
inference capabilities.

NOT FOR SAFETY-CRITICAL USE - Research and educational purposes only.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from pathlib import Path
import sys
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from utils.device_utils import set_deterministic_seed, get_device
from utils.config import SystemConfig, load_config
from pipelines.data_pipeline import SensorDataGenerator, SensorDataProcessor
from models.tensorflow_models import TensorFlowActivityCNN, EdgeOptimizedTensorFlowCNN
from models.pytorch_models import ActivityRecognitionCNN, EdgeOptimizedCNN
from export.edge_deployment import EdgeRuntime, PerformanceProfiler


# Page configuration
st.set_page_config(
    page_title="Smart Home Activity Recognition",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {}


def load_models():
    """Load pre-trained models."""
    try:
        # Load configuration
        config = load_config()
        
        # Set deterministic seed
        set_deterministic_seed(config.data.random_seed)
        
        models = {}
        
        # Try to load TensorFlow models
        try:
            tf_standard_path = "models/tensorflow_standard_model.h5"
            if Path(tf_standard_path).exists():
                models['tensorflow_standard'] = tf.keras.models.load_model(tf_standard_path)
            
            tf_edge_path = "models/tensorflow_edge_optimized_model.h5"
            if Path(tf_edge_path).exists():
                models['tensorflow_edge'] = tf.keras.models.load_model(tf_edge_path)
        except Exception as e:
            st.warning(f"Could not load TensorFlow models: {e}")
        
        # Try to load PyTorch models
        try:
            pytorch_standard_path = "models/pytorch_standard_model.pth"
            if Path(pytorch_standard_path).exists():
                model = ActivityRecognitionCNN(config.model)
                model.load_state_dict(torch.load(pytorch_standard_path, map_location='cpu'))
                model.eval()
                models['pytorch_standard'] = model
            
            pytorch_edge_path = "models/pytorch_edge_optimized_model.pth"
            if Path(pytorch_edge_path).exists():
                model = EdgeOptimizedCNN(config.model)
                model.load_state_dict(torch.load(pytorch_edge_path, map_location='cpu'))
                model.eval()
                models['pytorch_edge'] = model
        except Exception as e:
            st.warning(f"Could not load PyTorch models: {e}")
        
        return models, config
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, None


def generate_sensor_data(activity: str, config: SystemConfig, noise_level: float = 0.1) -> np.ndarray:
    """Generate synthetic sensor data for a specific activity.
    
    Args:
        activity: Activity name
        config: System configuration
        noise_level: Noise level for data generation
        
    Returns:
        Generated sensor data
    """
    generator = SensorDataGenerator(config.data)
    data, _ = generator.generate_activity_data(activity, 1, add_noise=True)
    
    # Add additional noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, data.shape)
        data = data + noise
    
    return data[0]  # Return single sample


def predict_activity(data: np.ndarray, model, model_type: str) -> tuple:
    """Predict activity from sensor data.
    
    Args:
        data: Sensor data
        model: Trained model
        model_type: Type of model
        
    Returns:
        Tuple of (predicted_class, confidence_scores)
    """
    if model_type.startswith('tensorflow'):
        # TensorFlow prediction
        prediction = model.predict(data.reshape(1, -1, data.shape[-1]), verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence_scores = prediction[0]
    else:
        # PyTorch prediction
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).unsqueeze(0)
            prediction = model(data_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()
            confidence_scores = torch.softmax(prediction, dim=1).numpy()[0]
    
    return predicted_class, confidence_scores


def create_sensor_visualization(data: np.ndarray, config: SystemConfig) -> go.Figure:
    """Create interactive visualization of sensor data.
    
    Args:
        data: Sensor data
        config: System configuration
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=config.data.num_sensors, cols=1,
        subplot_titles=[f"Sensor {i+1}" for i in range(config.data.num_sensors)],
        vertical_spacing=0.1
    )
    
    time_steps = np.arange(config.data.sequence_length)
    
    for i in range(config.data.num_sensors):
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=data[:, i],
                mode='lines+markers',
                name=f'Sensor {i+1}',
                line=dict(width=2),
                marker=dict(size=4)
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        title="Real-time Sensor Data",
        height=400,
        showlegend=False,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Time Steps")
    fig.update_yaxes(title_text="Sensor Value")
    
    return fig


def create_confidence_plot(confidence_scores: np.ndarray, activities: list) -> go.Figure:
    """Create confidence score visualization.
    
    Args:
        confidence_scores: Confidence scores for each class
        activities: List of activity names
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[
        go.Bar(
            x=activities,
            y=confidence_scores,
            marker_color=['#1f77b4' if i == np.argmax(confidence_scores) else '#d3d3d3' 
                         for i in range(len(activities))]
        )
    ])
    
    fig.update_layout(
        title="Activity Recognition Confidence",
        xaxis_title="Activities",
        yaxis_title="Confidence Score",
        template="plotly_white",
        height=400
    )
    
    return fig


def create_performance_dashboard(metrics: dict) -> None:
    """Create performance metrics dashboard.
    
    Args:
        metrics: Performance metrics dictionary
    """
    st.subheader("Performance Metrics")
    
    if not metrics:
        st.info("No performance metrics available. Run inference to see metrics.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Latency", f"{metrics.get('latency_ms', {}).get('mean', 0):.2f} ms")
    
    with col2:
        st.metric("Throughput", f"{metrics.get('throughput_fps', 0):.1f} FPS")
    
    with col3:
        st.metric("Memory Usage", f"{metrics.get('memory_mb', {}).get('mean', 0):.2f} MB")
    
    with col4:
        st.metric("Model Size", f"{metrics.get('model_size_mb', 0):.2f} MB")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏠 Smart Home Activity Recognition</h1>', 
                unsafe_allow_html=True)
    
    # Warning disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
        This system is for research and educational purposes only. 
        <strong>NOT FOR SAFETY-CRITICAL USE.</strong> 
        Do not rely on this system for any safety-critical applications.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Load models
    if not st.session_state.model_loaded:
        with st.spinner("Loading models..."):
            models, config = load_models()
            if models and config:
                st.session_state.models = models
                st.session_state.config = config
                st.session_state.model_loaded = True
                st.success("Models loaded successfully!")
            else:
                st.error("Failed to load models. Please ensure models are trained first.")
                st.stop()
    
    models = st.session_state.models
    config = st.session_state.config
    
    # Model selection
    available_models = list(models.keys())
    if not available_models:
        st.error("No models available. Please train models first.")
        st.stop()
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        help="Choose which model to use for inference"
    )
    
    # Activity selection
    activity = st.sidebar.selectbox(
        "Simulate Activity",
        config.data.activities,
        help="Select which activity to simulate"
    )
    
    # Noise level
    noise_level = st.sidebar.slider(
        "Noise Level",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
        help="Add noise to simulate real-world conditions"
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Real-time Sensor Data")
        
        # Generate sensor data
        sensor_data = generate_sensor_data(activity, config, noise_level)
        
        # Create visualization
        fig_sensors = create_sensor_visualization(sensor_data, config)
        st.plotly_chart(fig_sensors, use_container_width=True)
    
    with col2:
        st.subheader("Activity Recognition")
        
        # Predict button
        if st.button("🔍 Predict Activity", type="primary"):
            with st.spinner("Running inference..."):
                start_time = time.time()
                
                # Make prediction
                predicted_class, confidence_scores = predict_activity(
                    sensor_data, models[selected_model], selected_model
                )
                
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Update session state
                st.session_state.current_prediction = {
                    'true_activity': activity,
                    'predicted_activity': config.data.activities[predicted_class],
                    'confidence_scores': confidence_scores,
                    'inference_time': inference_time,
                    'timestamp': time.time()
                }
                
                # Add to history
                st.session_state.prediction_history.append(st.session_state.current_prediction)
        
        # Display current prediction
        if st.session_state.current_prediction:
            pred = st.session_state.current_prediction
            
            st.markdown("### Prediction Results")
            
            # True vs Predicted
            col_true, col_pred = st.columns(2)
            with col_true:
                st.markdown(f"**True Activity:** {pred['true_activity']}")
            with col_pred:
                st.markdown(f"**Predicted:** {pred['predicted_activity']}")
            
            # Confidence scores
            fig_conf = create_confidence_plot(pred['confidence_scores'], config.data.activities)
            st.plotly_chart(fig_conf, use_container_width=True)
            
            # Inference time
            st.metric("Inference Time", f"{pred['inference_time']:.2f} ms")
    
    # Performance dashboard
    st.subheader("Performance Dashboard")
    
    # Calculate performance metrics
    if st.session_state.prediction_history:
        inference_times = [p['inference_time'] for p in st.session_state.prediction_history]
        
        performance_metrics = {
            'latency_ms': {
                'mean': np.mean(inference_times),
                'std': np.std(inference_times),
                'min': np.min(inference_times),
                'max': np.max(inference_times)
            },
            'throughput_fps': 1000 / np.mean(inference_times),
            'num_predictions': len(inference_times)
        }
        
        create_performance_dashboard(performance_metrics)
    
    # Prediction history
    if st.session_state.prediction_history:
        st.subheader("Prediction History")
        
        # Create history dataframe
        history_data = []
        for pred in st.session_state.prediction_history[-10:]:  # Show last 10
            history_data.append({
                'Timestamp': time.strftime('%H:%M:%S', time.localtime(pred['timestamp'])),
                'True Activity': pred['true_activity'],
                'Predicted Activity': pred['predicted_activity'],
                'Confidence': f"{np.max(pred['confidence_scores']):.3f}",
                'Inference Time (ms)': f"{pred['inference_time']:.2f}"
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True)
        
        # Accuracy over time
        if len(st.session_state.prediction_history) > 1:
            correct_predictions = [
                1 if p['true_activity'] == p['predicted_activity'] else 0 
                for p in st.session_state.prediction_history
            ]
            
            cumulative_accuracy = np.cumsum(correct_predictions) / np.arange(1, len(correct_predictions) + 1)
            
            fig_accuracy = go.Figure(data=[
                go.Scatter(
                    x=list(range(len(cumulative_accuracy))),
                    y=cumulative_accuracy,
                    mode='lines+markers',
                    name='Cumulative Accuracy'
                )
            ])
            
            fig_accuracy.update_layout(
                title="Accuracy Over Time",
                xaxis_title="Number of Predictions",
                yaxis_title="Cumulative Accuracy",
                template="plotly_white",
                height=300
            )
            
            st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # Model information
    st.subheader("Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Selected Model:** {selected_model}")
        st.markdown(f"**Framework:** {'TensorFlow' if 'tensorflow' in selected_model else 'PyTorch'}")
        st.markdown(f"**Type:** {'Edge Optimized' if 'edge' in selected_model else 'Standard'}")
    
    with col2:
        st.markdown(f"**Sequence Length:** {config.data.sequence_length}")
        st.markdown(f"**Number of Sensors:** {config.data.num_sensors}")
        st.markdown(f"**Activities:** {', '.join(config.data.activities)}")
    
    with col3:
        st.markdown(f"**Device:** {get_device()}")
        st.markdown(f"**Noise Level:** {noise_level:.2f}")
        st.markdown(f"**Total Predictions:** {len(st.session_state.prediction_history)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Smart Home Activity Recognition Demo | 
        <strong>NOT FOR SAFETY-CRITICAL USE</strong> | 
        Research and Educational Purposes Only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
