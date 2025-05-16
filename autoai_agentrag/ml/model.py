"""
Machine Learning Model base implementation for AutoAI-AgentRAG.

This module provides the base class and utilities for integrating machine learning
models (TensorFlow, PyTorch, etc.) into the AutoAI-AgentRAG framework.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, cast

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Type variable for the MLModel class to use in classmethods
T = TypeVar('T', bound='MLModel')


class ModelFramework(str, Enum):
    """Enumeration of supported ML frameworks."""
    
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"
    ONNX = "onnx"


class ModelMetadata(BaseModel):
    """Metadata for machine learning models."""
    
    name: str = Field(..., description="Name of the model")
    version: str = Field("0.1.0", description="Version of the model")
    description: Optional[str] = Field(None, description="Description of what the model does")
    framework: ModelFramework = Field(..., description="ML framework used for this model")
    created_at: datetime = Field(default_factory=datetime.now, description="When the model was created")
    last_modified: datetime = Field(default_factory=datetime.now, description="When the model was last modified")
    author: Optional[str] = Field(None, description="Author of the model")
    tags: List[str] = Field(default_factory=list, description="Tags associated with this model")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Schema for model inputs")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Schema for model outputs")
    
    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class ModelConfig(BaseModel):
    """Configuration parameters for ML models."""
    
    batch_size: int = Field(default=32, description="Batch size for inference")
    device: str = Field(default="auto", description="Device to run model on (cpu, cuda, auto)")
    precision: str = Field(default="float32", description="Numerical precision for model computation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens for text generation models")
    timeout_seconds: int = Field(default=30, description="Timeout for model inference in seconds")
    cache_dir: Optional[str] = Field(default=None, description="Directory to cache model files")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "allow"


class MLModel(ABC):
    """
    Base class for machine learning models in AutoAI-AgentRAG.
    
    This abstract class defines the interface for all ML models integrated
    into the framework, regardless of the underlying framework (TensorFlow,
    PyTorch, etc.).
    
    Attributes:
        name (str): Name of the model
        metadata (ModelMetadata): Metadata for this model
        config (ModelConfig): Configuration for this model
        is_loaded (bool): Whether the model is loaded and ready for inference
    """
    
    def __init__(
        self, 
        name: str,
        framework: ModelFramework = ModelFramework.CUSTOM,
        config: Optional[ModelConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new ML model.
        
        Args:
            name: Name of the model
            framework: ML framework used by this model
            config: Optional configuration parameters
            metadata: Optional metadata for the model
        """
        self.name = name
        self.config = config or ModelConfig()
        
        # Initialize metadata
        meta_dict = metadata or {}
        meta_dict.update({
            "name": name,
            "framework": framework
        })
        self.metadata = ModelMetadata(**meta_dict)
        
        self.is_loaded = False
        self._model = None
        
        logger.info(f"ML Model '{name}' initialized with {framework} framework")
    
    @abstractmethod
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load the model from disk or initialize it.
        
        Args:
            model_path: Optional path to model weights or definition
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference on the model with the given inputs.
        
        Args:
            inputs: Input data for model inference
            **kwargs: Additional inference parameters
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def save(self, save_path: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            save_path: Path to save the model to
            
        Returns:
            True if the model was saved successfully, False otherwise
        """
        pass
    
    def save_metadata(self, save_path: Optional[str] = None) -> bool:
        """
        Save the model metadata to disk.
        
        Args:
            save_path: Path to save the metadata to. If None, uses model_path + '.meta.json'
            
        Returns:
            True if metadata was saved successfully, False otherwise
        """
        try:
            path = save_path or f"{self.name}.meta.json"
            
            # Update last_modified timestamp
            self.metadata.last_modified = datetime.now()
            
            with open(path, "w") as f:
                json.dump(self.metadata.dict(), f, indent=2)
                
            logger.info(f"Model metadata saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}")
            return False
    
    @classmethod
    def load_metadata(cls, metadata_path: str) -> Optional[ModelMetadata]:
        """
        Load model metadata from disk.
        
        Args:
            metadata_path: Path to the metadata file
            
        Returns:
            The loaded metadata, or None if loading failed
        """
        try:
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
            
            return ModelMetadata(**metadata_dict)
        except Exception as e:
            logger.error(f"Error loading model metadata: {str(e)}")
            return None
    
    @classmethod
    def from_pretrained(cls: type[T], model_name_or_path: str, **kwargs) -> T:
        """
        Create a model from a pretrained checkpoint or model repository.
        
        Args:
            model_name_or_path: Name of a pretrained model or path to model weights
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            An initialized MLModel instance
        """
        # Infer the framework from the model file or name
        framework = cls._infer_framework(model_name_or_path)
        
        if framework == ModelFramework.TENSORFLOW:
            # Dynamic import to avoid TensorFlow dependency when not needed
            try:
                from autoai_agentrag.ml.tensorflow import TensorFlowModel
                model_cls = TensorFlowModel
            except ImportError:
                logger.warning("TensorFlow not installed. Using CustomModel as fallback.")
                model_cls = CustomModel
        elif framework == ModelFramework.PYTORCH:
            # Dynamic import to avoid PyTorch dependency when not needed
            try:
                from autoai_agentrag.ml.pytorch import PyTorchModel
                model_cls = PyTorchModel
            except ImportError:
                logger.warning("PyTorch not installed. Using CustomModel as fallback.")
                model_cls = CustomModel
        elif framework == ModelFramework.HUGGINGFACE:
            # Dynamic import to avoid Hugging Face dependency when not needed
            try:
                from autoai_agentrag.ml.huggingface import HuggingFaceModel
                model_cls = HuggingFaceModel
            except ImportError:
                logger.warning("Hugging Face Transformers not installed. Using CustomModel as fallback.")
                model_cls = CustomModel
        else:
            logger.warning(f"Could not infer framework for {model_name_or_path}, defaulting to custom model")
            model_cls = CustomModel
        
        # Extract model name from path if needed
        model_name = os.path.basename(model_name_or_path).split(".")[0]
        
        # Create the model instance
        model = model_cls(name=model_name, framework=framework, **kwargs)
        
        # Load the model
        model.load(model_name_or_path)
        return cast(T, model)
    
    @staticmethod
    def _infer_framework(model_path: str) -> ModelFramework:
        """
        Infer the ML framework based on the model file or name.
        
        Args:
            model_path: Path to model file or name of model
            
        Returns:
            The inferred ModelFramework
        """
        if os.path.exists(model_path):
            # Check file extension
            if model_path.endswith((".h5", ".keras", ".tf", ".pb")):
                return ModelFramework.TENSORFLOW
            elif model_path.endswith((".pt", ".pth", ".pkl")):
                return ModelFramework.PYTORCH
            elif model_path.endswith((".onnx")):
                return ModelFramework.ONNX
            elif model_path.endswith((".joblib", ".pickle")):
                return ModelFramework.SKLEARN
        
        # Check if it's a HuggingFace model identifier
        if "/" in model_path or model_path.startswith("hf://"):
            return ModelFramework.HUGGINGFACE
        
        # Default to custom
        return ModelFramework.CUSTOM
    
    def __call__(self, inputs: Any, **kwargs) -> Any:
        """
        Call method to make the model directly callable.
        
        Args:
            inputs: Input data for model inference
            **kwargs: Additional inference parameters
            
        Returns:
            Model predictions
        """
        return self.predict(inputs, **kwargs)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "loaded" if self.is_loaded else "not loaded"
        return f"MLModel(name='{self.name}', framework={self.metadata.framework}, status={status})"


class CustomModel(MLModel):
    """
    A custom ML model implementation that can wrap arbitrary model objects.
    
    This class provides a way to integrate custom or third-party model implementations
    that don't fit into the standard frameworks.
    """
    
    def __init__(
        self,
        name: str,
        framework: ModelFramework = ModelFramework.CUSTOM,
        config: Optional[ModelConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        predict_fn: Optional[callable] = None
    ):
        """
        Initialize a new custom model.
        
        Args:
            name: Name of the model
            framework: ML framework used by this model
            config: Optional configuration parameters
            metadata: Optional metadata for the model
            predict_fn: Optional custom predict function to use instead of the model's
        """
        super().__init__(name, framework, config, metadata)
        self._predict_fn = predict_fn
    
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load a custom model from disk or initialize it.
        
        Args:
            model_path: Path to model file or directory
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            if model_path and os.path.exists(model_path):
                # Try to load with pickle or other serialization methods
                if model_path.endswith((".pkl", ".pickle")):
                    import pickle
                    with open(model_path, "rb") as f:
                        self._model = pickle.load(f)
                elif model_path.endswith(".joblib"):
                    from joblib import load
                    self._model = load(model_path)
                else:
                    logger.warning(f"Unsupported file format for {model_path}. Loading as pickle.")
                    import pickle
                    with open(model_path, "rb") as f:
                        self._model = pickle.load(f)
                
                # Look for metadata file
                meta_path = f"{model_path}.meta.json"
                if os.path.exists(meta_path):
                    try:
                        metadata = self.load_metadata(meta_path)
                        if metadata:
                            self.metadata = metadata
                    except Exception as e:
                        logger.warning(f"Could not load metadata from {meta_path}: {e}")
                
                self.is_loaded = True
                logger.info(f"Custom model '{self.name}' loaded from {model_path}")
                return True
            else:
                logger.warning(f"No model file found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading custom model: {str(e)}")
            return False
    
    def predict(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference on the custom model.
        
        Args:
            inputs: Input data for model inference
            **kwargs: Additional inference parameters
            
        Returns:
            Model predictions
        """
        if not self.is_loaded:
            raise ValueError("Model is not loaded. Call load() before predict().")
        
        try:
            start_time = time.time()
            
            # Use custom predict function if provided
            if self._predict_fn is not None:
                result = self._predict_fn(inputs, **kwargs)
            # Otherwise try to use the model's predict method
            elif hasattr(self._model, "predict"):
                result = self._model.predict(inputs, **kwargs)
            # Or call the model directly if it's callable
            elif callable(self._model):
                result = self._model(inputs, **kwargs)
            else:
                raise ValueError("Model does not have a predict method and is not callable.")
            
            # Calculate and log inference time
            inference_time = time.time() - start_time
            logger.debug(f"Inference time: {inference_time:.4f}s")
            
            return result
        except Exception as e:
            logger.error(f"Error during model inference: {str(e)}")
            raise
    
    def save(self, save_path: str) -> bool:
        """
        Save the custom model to disk.
        
        Args:
            save_path: Path to save the model to
            
        Returns:
            True if model was saved successfully, False otherwise
        """
        if not self.is_loaded or self._model is None:
            logger.warning("Cannot save model that isn't loaded")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save using appropriate method based on file extension
            if save_path.endswith((".pkl", ".pickle")):
                import pickle
                with open(save_path, "wb") as f:
                    pickle.dump(self._model, f)
            elif save_path.endswith(".

"""
Machine Learning Model base implementation for AutoAI-AgentRAG.

This module provides the base class and utilities for integrating machine learning
models (TensorFlow, PyTorch, etc.) into the AutoAI-AgentRAG framework.
"""

import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelFramework(str, Enum):
    """Enumeration of supported ML frameworks."""
    
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class ModelConfig(BaseModel):
    """Configuration parameters for ML models."""
    
    batch_size: int = Field(default=32, description="Batch size for inference")
    device: str = Field(default="auto", description="Device to run model on (cpu, cuda, auto)")
    precision: str = Field(default="float32", description="Numerical precision for model computation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens for text generation models")
    timeout_seconds: int = Field(default=30, description="Timeout for model inference in seconds")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "allow"


class MLModel(ABC):
    """
    Base class for machine learning models in AutoAI-AgentRAG.
    
    This abstract class defines the interface for all ML models integrated
    into the framework, regardless of the underlying framework (TensorFlow,
    PyTorch, etc.).
    
    Attributes:
        name (str): Name of the model
        framework (ModelFramework): ML framework used by this model
        config (ModelConfig): Configuration for this model
        is_loaded (bool): Whether the model is loaded and ready for inference
    """
    
    def __init__(
        self, 
        name: str,
        framework: ModelFramework = ModelFramework.CUSTOM,
        config: Optional[ModelConfig] = None
    ):
        """
        Initialize a new ML model.
        
        Args:
            name: Name of the model
            framework: ML framework used by this model
            config: Optional configuration parameters
        """
        self.name = name
        self.framework = framework
        self.config = config or ModelConfig()
        self.is_loaded = False
        
        self._model = None
        logger.info(f"ML Model '{name}' initialized with {framework} framework")
    
    @abstractmethod
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load the model from disk or initialize it.
        
        Args:
            model_path: Optional path to model weights or definition
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """
        Run inference on the model with the given inputs.
        
        Args:
            inputs: Input data for model inference
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def save(self, save_path: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            save_path: Path to save the model to
            
        Returns:
            True if the model was saved successfully, False otherwise
        """
        pass
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> 'MLModel':
        """
        Create a model from a pretrained checkpoint or model repository.
        
        Args:
            model_name_or_path: Name of a pretrained model or path to model weights
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            An initialized MLModel instance
        """
        # Infer the framework from the model file or name
        framework = cls._infer_framework(model_name_or_path)
        
        if framework == ModelFramework.TENSORFLOW:
            from autoai_agentrag.ml.tensorflow import TensorFlowModel
            model = TensorFlowModel(name=model_name_or_path.split("/")[-1], **kwargs)
        elif framework == ModelFramework.PYTORCH:
            from autoai_agentrag.ml.pytorch import PyTorchModel
            model = PyTorchModel(name=model_name_or_path.split("/")[-1], **kwargs)
        elif framework == ModelFramework.HUGGINGFACE:
            from autoai_agentrag.ml.huggingface import HuggingFaceModel
            model = HuggingFaceModel(name=model_name_or_path, **kwargs)
        else:
            logger.warning(f"Could not infer framework for {model_name_or_path}, defaulting to custom model")
            model = CustomModel(name=model_name_or_path.split("/")[-1], **kwargs)
        
        # Load the model
        model.load(model_name_or_path)
        return model
    
    @staticmethod
    def _infer_framework(model_path: str) -> ModelFramework:
        """
        Infer the ML framework based on the model file or name.
        
        Args:
            model_path: Path to model file or name of model
            
        Returns:
            The inferred ModelFramework
        """
        if os.path.exists(model_path):
            # Check file extension
            if model_path.endswith(".h5") or model_path.endswith(".keras") or model_path.endswith(".tf"):
                return ModelFramework.TENSORFLOW
            elif model_path.endswith(".pt") or model_path.endswith(".pth"):
                return ModelFramework.PYTORCH
            elif model_path.endswith(".pkl") or model_path.endswith(".joblib"):
                return ModelFramework.SKLEARN
        
        # Check if it's a HuggingFace model name
        if "/" in model_path or model_path.startswith("hf://"):
            return ModelFramework.HUGGINGFACE
        
        # Default to custom
        return ModelFramework.CUSTOM
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "loaded" if self.is_loaded else "not loaded"
        return f"MLModel(name='{self.name}', framework={self.framework}, status={status})"


class CustomModel(MLModel):
    """
    A custom ML model implementation that can wrap arbitrary model objects.
    
    This class provides a way to integrate custom or third-party model implementations
    that don't fit into the standard frameworks.
    """
    
    def __init__(
        self,
        name: str,
        framework: ModelFramework = ModelFramework.CUSTOM,
        config: Optional[ModelConfig] = None,
        predict_fn: Optional[callable] = None
    ):
        """
        Initialize a new custom model.
        
        Args:
            name: Name of the model
            framework: ML framework used by this model
            config: Optional configuration parameters
            predict_fn: Optional custom predict function to use instead of the model's
        """
        super().__init__(name, framework, config)
        self._predict_fn = predict_fn
    
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load a custom model from disk or initialize it.
        
        Args:
            model_path: Path to model file or directory
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            if model_path and os.path.exists(model_path):
                # Try to load with pickle
                import pickle
                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
                    
                self.is_loaded = True
                logger.info(f"Custom model '{self.name}' loaded from {model_path}")
                return True
            else:
                logger.warning(f"No model file found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading custom model: {str(e)}")
            return False
    
    def predict(self, inputs: Any) -> Any:
        """
        Run inference on the custom model.
        
        Args:
            inputs: Input data for model inference
            
        Returns:
            Model predictions
        """
        if not self.is_loaded:
            raise ValueError("Model is not loaded. Call load() before predict().")
        
        if self._predict_fn is not None:
            return self._predict_fn(inputs)
        
        if hasattr(self._model, "predict"):
            return self._model.predict(inputs)
        elif callable(self._model):
            return self._model(inputs)
        else:
            raise ValueError("Model does not have a predict method and is not callable.")
    
    def save(self, save_path: str) -> bool:
        """
        Save the custom model to disk.
        
        Args:
            save_path: Path to save the model to
            
        Returns:
            True if model was saved successfully, False otherwise
        """
        if not self.is_loaded:
            logger.warning("Cannot save model that isn't loaded")
            return False
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Try to save with pickle
            import pickle
            with open(save_path, "wb") as f:
                pickle.dump(self._model, f)
                
            logger.info(f"Custom model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving custom model: {str(e)}")
            return False

