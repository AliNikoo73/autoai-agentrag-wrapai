#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML Models Example

This example demonstrates how to use different machine learning models 
(TensorFlow, PyTorch) with AutoAI-AgentRAG. It shows how to load, configure
and integrate models with agents for inference.
"""

import logging
import sys
import os
import numpy as np
import tempfile
from typing import Dict, Any, Union, Optional

from autoai_agentrag import Agent
from autoai_agentrag.ml.model import MLModel, ModelConfig, ModelFramework, ModelMetadata, CustomModel

# Try to import framework-specific models
try:
    from autoai_agentrag.ml.tensorflow import TensorFlowModel, TensorFlowConfig
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from autoai_agentrag.ml.pytorch import PyTorchModel, PyTorchConfig
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class SimpleMockModel:
    """A simple mock ML model for demonstration purposes."""
    
    def __init__(self, name: str):
        """Initialize the mock model."""
        self.name = name
        
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Mock prediction function."""
        # Just return a simple transformation of the input
        return inputs * 2 + 1


def create_mock_model_file() -> str:
    """Create a mock model file for demonstration purposes."""
    import pickle
    
    # Create a simple model
    model = SimpleMockModel("demo-model")
    
    # Save it to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(model, f)
        temp_path = f.name
        
    return temp_path


def main():
    """
    Demonstrate different ML model integrations.
    
    This function shows how to:
    1. Create and use different types of ML models
    2. Load models from files
    3. Perform inference using models
    4. Connect models to agents
    """
    try:
        # Create a mock model file
        model_path = create_mock_model_file()
        logger.info(f"Created mock model file at: {model_path}")
        
        # Load the model using CustomModel (works without TensorFlow/PyTorch)
        logger.info("\nLoading model with CustomModel")
        custom_model = CustomModel(name="custom-demo-model")
        
        if custom_model.load(model_path):
            logger.info("Model loaded successfully")
            
            # Make predictions
            input_data = np.array([1, 2, 3, 4, 5])
            logger.info(f"Input data: {input_data}")
            
            result = custom_model.predict(input_data)
            logger.info(f"Prediction result: {result}")
            
            # Define a custom prediction function
            def custom_predict_fn(inputs):
                return inputs * 3 + 2
                
            # Create a model with custom prediction function
            logger.info("\nCreating a model with custom prediction function")
            fn_model = CustomModel(
                name="function-model", 
                predict_fn=custom_predict_fn
            )
            fn_model.is_loaded = True  # Mark as loaded since we're using a function
            
            result = fn_model.predict(input_data)
            logger.info(f"Custom function prediction: {result}")
        else:
            logger.error("Failed to load model")
            
        # Try framework-specific models if available
        if TENSORFLOW_AVAILABLE:
            logger.info("\nTensorFlow is available, demonstrating TensorFlowModel")
            # In a real scenario, you would use actual TensorFlow model files
            # This is just a placeholder example
            tf_config = TensorFlowConfig(
                eager_mode=True,
                mixed_precision=False,
                dynamic_shape=True
            )
            tf_model = TensorFlowModel(name="tf-demo", config=tf_config)
            logger.info(f"Created TensorFlowModel with config: {tf_config}")
        else:
            logger.info("\nTensorFlow is not available, skipping TensorFlowModel example")
            
        if PYTORCH_AVAILABLE:
            logger.info("\nPyTorch is available, demonstrating PyTorchModel")
            # In a real scenario, you would use actual PyTorch model files
            # This is just a placeholder example
            pt_config = PyTorchConfig(
                use_jit=False,
                use_fp16=False,
                device="cpu"
            )
            pt_model = PyTorchModel(name="pytorch-demo", config=pt_config)
            logger.info(f"Created PyTorchModel with config: {pt_config}")
        else:
            logger.info("\nPyTorch is not available, skipping PyTorchModel example")
            
        # Connect a model to an agent
        logger.info("\nConnecting model to an agent")
        agent = Agent(name="ml-enabled-agent")
        agent.add_model(custom_model)
        
        # Execute a task that can use the model
        logger.info("Executing a task with the ML-enabled agent")
        result = agent.execute("Process this with the model")
        
        # Display the result
        if result.success:
            logger.info(f"Task completed successfully")
            if "models_used" in result.result:
                logger.info(f"Models used: {result.result['models_used']}")
        else:
            logger.error(f"Task failed: {result.error}")
            
        # Clean up the temporary file
        try:
            os.remove(model_path)
            logger.info(f"Removed temporary model file: {model_path}")
        except OSError as e:
            logger.warning(f"Failed to remove temporary file: {str(e)}")
            
        logger.info("\nML models demonstration completed")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

