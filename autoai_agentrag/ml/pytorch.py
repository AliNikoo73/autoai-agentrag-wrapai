"""
PyTorch model implementation for AutoAI-AgentRAG.

This module provides a PyTorch-specific implementation of the MLModel class
for integrating PyTorch models into the AutoAI-AgentRAG framework.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from autoai_agentrag.ml.model import MLModel, ModelConfig, ModelFramework, ModelMetadata

logger = logging.getLogger(__name__)

# Try to import PyTorch, but don't fail if it's not installed
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. PyTorchModel will not be functional.")
    PYTORCH_AVAILABLE = False
    # Create a placeholder for type hints
    class torch:
        class nn:
            class Module:
                pass
        Tensor = Any
        jit = Any
        device = Any


class PyTorchConfig(ModelConfig):
    """PyTorch-specific configuration parameters."""
    
    use_jit: bool = False
    use_fp16: bool = False
    use_cuda: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    model_format: str = "pt"  # "pt", "script", "onnx"
    trace_model: bool = False
    num_workers: int = 4
    pin_memory: bool = True


class PyTorchModel(MLModel):
    """
    PyTorch-specific implementation of the MLModel class.
    
    This class provides PyTorch-specific methods for loading, saving, and
    running inference with PyTorch models.
    
    Attributes:
        name (str): Name of the model
        metadata (ModelMetadata): Metadata for this model
        config (PyTorchConfig): PyTorch-specific configuration
        is_loaded (bool): Whether the model is loaded and ready for inference
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[PyTorchConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new PyTorch model.
        
        Args:
            name: Name of the model
            config: Optional PyTorch-specific configuration
            metadata: Optional metadata for the model
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Please install PyTorch to use PyTorchModel.")
        
        super().__init__(
            name=name,
            framework=ModelFramework.PYTORCH,
            config=config or PyTorchConfig(),
            metadata=metadata
        )
        
        # Cast config to PyTorchConfig for type hints
        self.config = cast(PyTorchConfig, self.config)
        
        # Configure PyTorch based on the config
        self._configure_pytorch()
        
        # Initialize model attribute
        self._model: Optional[torch.nn.Module] = None
        self._device = None  # Will be set during configuration
        
    def _configure_pytorch(self):
        """Configure PyTorch based on the model configuration."""
        if not PYTORCH_AVAILABLE:
            return
        
        # Set cuDNN behavior
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
            torch.backends.cudnn.deterministic = self.config.cudnn_deterministic
            
        # Set device
        if self.config.device != "auto":
            if self.config.device == "cpu":
                self._device = torch.device("cpu")
                logger.info("PyTorch configured to use CPU")
            elif self.config.device.startswith("cuda") or self.config.device.startswith("gpu"):
                # Extract device index if specified
                if ":" in self.config.device:
                    device_idx = int(self.config.device.split(":")[-1])
                    device_str = f"cuda:{device_idx}"
                else:
                    device_str = "cuda:0"  # Default to first GPU
                
                if torch.cuda.is_available():
                    self._device = torch.device(device_str)
                    logger.info(f"PyTorch configured to use {device_str}")
                else:
                    logger.warning("CUDA requested but not available. Using CPU instead.")
                    self._device = torch.device("cpu")
            else:
                logger.warning(f"Unknown device '{self.config.device}'. Using auto-detection.")
                self._device = torch.device("cuda:0" if torch.cuda.is_available() and self.config.use_cuda else "cpu")
        else:
            # Auto-detect best device
            self._device = torch.device("cuda:0" if torch.cuda.is_available() and self.config.use_cuda else "cpu")
            logger.info(f"PyTorch using device: {self._device}")
        
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load a PyTorch model from disk.
        
        Args:
            model_path: Path to the PyTorch model file
            
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch is not installed. Cannot load PyTorch model.")
            return False
        
        if model_path is None:
            logger.error("Model path is required to load a PyTorch model")
            return False
        
        try:
            start_time = time.time()
            
            # Determine model format from file extension or config
            is_jit = False
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                # Regular PyTorch checkpoint or TorchScript
                # Try to load as TorchScript first (JIT model)
                try:
                    if self.config.use_jit:
                        self._model = torch.jit.load(model_path, map_location=self._device)
                        is_jit = True
                        logger.info(f"Loaded TorchScript model from {model_path}")
                    else:
                        # Load as regular checkpoint
                        checkpoint = torch.load(model_path, map_location=self._device)
                        
                        # Check if it's just the state dict or a full checkpoint
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            # This is a full checkpoint with state_dict
                            # We need an actual model instance to load into
                            if hasattr(self, 'create_model'):
                                # If the subclass provides a model factory method, use it
                                self._model = self.create_model()
                                self._model.load_state_dict(checkpoint['state_dict'])
                                
                                # Extract metadata if available
                                if 'metadata' in checkpoint and isinstance(checkpoint['metadata'], dict):
                                    self.metadata = ModelMetadata(**checkpoint['metadata'])
                            else:
                                logger.error("Checkpoint contains state_dict but no model factory method available")
                                return False
                        elif isinstance(checkpoint, dict) and not 'state_dict' in checkpoint:
                            # Assume it's just a state_dict
                            if hasattr(self, 'create_model'):
                                self._model = self.create_model()
                                self._model.load_state_dict(checkpoint)
                            else:
                                logger.error("State dict loaded but no model factory method available")
                                return False
                        elif isinstance(checkpoint, torch.nn.Module):
                            # It's a full model
                            self._model = checkpoint
                        else:
                            logger.error(f"Unknown checkpoint format in {model_path}")
                            return False
                        
                        logger.info(f"Loaded PyTorch checkpoint from {model_path}")
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    return False
            elif model_path.endswith('.onnx'):
                logger.error("ONNX models not directly supported in PyTorchModel. Use ONNXModel instead.")
                return False
            else:
                logger.error(f"Unsupported PyTorch model format for {model_path}")
                return False
            
            # Ensure model is on the correct device
            if not is_jit and self._model is not None:
                self._model = self._model.to(self._device)
            
            # Set evaluation mode
            if hasattr(self._model, 'eval'):
                self._model.eval()
            
            # Convert to JIT if requested and not already JIT
            if self.config.use_jit and not is_jit and self._model is not None:
                try:
                    # Create a traced version of the model
                    dummy_input_shape = getattr(self, 'input_shape', (1, 3, 224, 224))
                    dummy_input = torch.rand(*dummy_input_shape).to(self._device)
                    self._model = torch.jit.trace(self._model, dummy_input)
                    logger.info("Model converted to TorchScript (JIT) format")
                except Exception as e:
                    logger.warning(f"Failed to convert model to TorchScript: {str(e)}")
            
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
            load_time = time.time() - start_time
            logger.info(f"PyTorch model '{self.name}' loaded in {load_time:.2f}s")
            
            # Log model summary if possible
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    model_params = sum(p.numel() for p in self._model.parameters())
                    logger.debug(f"Model parameters: {model_params:,}")
                except:
                    pass
                    
            return True
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {str(e)}")
            return False
    
    def predict(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference on the PyTorch model.
        
        Args:
            inputs: Input data for model inference (can be numpy array, tensor, or list)
            **kwargs: Additional inference parameters
            
        Returns:
            Model predictions (usually as numpy array)
        """
        if not self.is_loaded or self._model is None:
            raise ValueError("Model is not loaded. Call load() before predict().")
        
        try:
            start_time = time.time()
            
            # Convert inputs to PyTorch tensors if they aren't already
            if not isinstance(inputs, torch.Tensor):
                if isinstance(inputs, np.ndarray):
                    inputs = torch.tensor(inputs, device=self._device)
                elif isinstance(inputs, list):
                    inputs = torch.tensor(np.array(inputs), device=self._device)
                else:
                    raise ValueError(f"Unsupported input type: {type(inputs)}")
            
            # Ensure inputs are on the correct device
            if inputs.device != self._device:
                inputs = inputs.to(self._device)
            
            # Handle FP16 precision if configured
            if self.config.use_fp16 and self._device.type != 'cpu':
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        outputs = self._model(inputs, **kwargs)
            else:
                # Regular inference
                with torch.no_grad():
                    outputs = self._model(inputs, **kwargs)
            
            # Convert outputs to numpy if they're tensors
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.detach().cpu().numpy()
            elif isinstance(outputs, tuple) and all(isinstance(o, torch.Tensor) for o in outputs):
                outputs = tuple(o.detach().cpu().numpy() for o in outputs)
            
            # Calculate and log inference time
            inference_time = time.time() - start_time
            logger.debug(f"PyTorch inference time: {inference_time:.4f}s")
            
            return outputs
        except Exception as e:
            logger.error(f"Error during PyTorch model inference: {str(e)}")
            raise
    
    def save(self, save_path: str) -> bool:
        """
        Save the PyTorch model to disk.
        
        Args:
            save_path: Path to save the model to
            
        Returns:
            True if the model was saved successfully, False otherwise
        """
        if not self.is_loaded or self._model is None:
            logger.warning("Cannot save model that isn't loaded")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Check if it's a JIT model
            is_jit = isinstance(self._model, torch.jit.ScriptModule)
            
            # Save based on format
            if self.config.model_format == "script" or is_jit:
                # Save as TorchScript
                if not is_jit:
                    # Convert to JIT first
                    try:
                        dummy_input_shape = getattr(self, 'input_shape', (1, 3, 224, 224))
                        dummy_input = torch.rand(*dummy_input_shape).to(self._device)
                        scripted_model = torch.jit.trace(self._model, dummy_input)
                        scripted_model.save(save_path)
                    except Exception as e:
                        logger.error(f"Error converting to TorchScript: {str(e)}")
                        return False
                else:
                    # Already JIT model
                    self._model.save(save_path)
            elif self.config.model_format == "onnx":
                # Save as ONNX
                try:
                    import onnx
                    import onnxruntime
                    
                    dummy_input_shape = getattr(self, 'input_shape', (1, 3, 224, 224))
                    dummy_input = torch.rand(*dummy_input_shape).to(self._device)
                    
                    # Export to ONNX
                    torch.onnx.export(
                        self._model,
                        dummy_input,
                        save_path,
                        export_params=True,
                        opset_version=13,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'},
                                     'output': {0: 'batch_size'}}
                    )
                    
                    # Verify the ONNX model
                    onnx_model = onnx.load(save_path)
                    onnx.checker.check_model(onnx_model)
                    
                    logger.info(f"Model saved in ONNX format to {save_path} and verified")
                except ImportError:
                    logger.error("ONNX export requires onnx and onnxruntime packages")
                    return False
                except Exception as e:
                    logger.error(f"Error exporting to ONNX: {str(e)}")
                    return False
            else:
                # Default to standard PyTorch save
                save_dict = {
                    'state_dict': self._model.state_dict(),
                    'metadata': self.metadata.dict() if hasattr(self, 'metadata') else {}
                }
                torch.save(save_dict, save_path)
            
            # Save metadata
            meta_path = f"{save_path}.meta.json"
            self.save_metadata(meta_path)
            
            logger.info(f"PyTorch model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving PyTorch model: {str(e)}")
            

