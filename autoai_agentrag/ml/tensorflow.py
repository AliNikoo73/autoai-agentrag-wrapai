"""
TensorFlow model implementation for AutoAI-AgentRAG.

This module provides a TensorFlow-specific implementation of the MLModel class
for integrating TensorFlow models into the AutoAI-AgentRAG framework.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from autoai_agentrag.ml.model import MLModel, ModelConfig, ModelFramework, ModelMetadata

logger = logging.getLogger(__name__)

# Try to import TensorFlow, but don't fail if it's not installed
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not installed. TensorFlowModel will not be functional.")
    TENSORFLOW_AVAILABLE = False
    # Create a placeholder for type hints
    class tf:
        class keras:
            class Model:
                pass
            class Sequential:
                pass
        class saved_model:
            pass
        class lite:
            class Interpreter:
                pass
            class TFLiteConverter:
                pass


class TFModelFormat(str):
    """Supported TensorFlow model formats."""
    
    SAVED_MODEL = "saved_model"
    H5 = "h5"
    KERAS = "keras"
    TF_LITE = "tflite"
    TF_JS = "tfjs"


class TensorFlowConfig(ModelConfig):
    """TensorFlow-specific configuration parameters."""
    
    eager_mode: bool = True
    mixed_precision: bool = False
    xla_compilation: bool = False
    model_format: TFModelFormat = TFModelFormat.SAVED_MODEL
    dynamic_shape: bool = True
    use_tensorrt: bool = False
    tpu_name: Optional[str] = None  # For TPU support
    distribute_strategy: Optional[str] = None  # "mirrored", "tpu", "parameter_server", "multi_worker_mirrored"
    allow_soft_placement: bool = True
    enable_op_determinism: bool = False
    memory_growth: bool = True
    use_graph_optimization: bool = True


class TensorFlowModel(MLModel):
    """
    TensorFlow-specific implementation of the MLModel class.
    
    This class provides TensorFlow-specific methods for loading, saving, and
    running inference with TensorFlow models.
    
    Attributes:
        name (str): Name of the model
        metadata (ModelMetadata): Metadata for this model
        config (TensorFlowConfig): TensorFlow-specific configuration
        is_loaded (bool): Whether the model is loaded and ready for inference
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[TensorFlowConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new TensorFlow model.
        
        Args:
            name: Name of the model
            config: Optional TensorFlow-specific configuration
            metadata: Optional metadata for the model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Please install TensorFlow to use TensorFlowModel.")
        
        super().__init__(
            name=name,
            framework=ModelFramework.TENSORFLOW,
            config=config or TensorFlowConfig(),
            metadata=metadata
        )
        
        # Cast config to TensorFlowConfig for type hints
        self.config = cast(TensorFlowConfig, self.config)
        
        # Configure TensorFlow based on the config
        self._configure_tensorflow()
        
        # Initialize model attribute
        self._model = None
        self._signatures = {}
        self._distribution_strategy = None
        
    def _configure_tensorflow(self):
        """Configure TensorFlow based on the model configuration."""
        if not TENSORFLOW_AVAILABLE:
            return
        
        # Set eager execution mode
        tf.config.run_functions_eagerly(self.config.eager_mode)
        
        # Enable op determinism for reproducibility if requested
        if self.config.enable_op_determinism:
            try:
                tf.config.experimental.enable_op_determinism()
                logger.info("TensorFlow deterministic operations enabled")
            except AttributeError:
                logger.warning("TensorFlow version does not support enable_op_determinism")
        
        # Configure device memory growth to prevent TF from allocating all memory
        if self.config.memory_growth:
            try:
                for gpu in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("GPU memory growth enabled")
            except:
                logger.warning("Failed to set memory growth for GPUs")
        
        # Configure mixed precision if requested
        if self.config.mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision enabled for TensorFlow model")
            except:
                logger.warning("Failed to set mixed precision policy")
        
        # Configure XLA compilation
        if self.config.xla_compilation:
            try:
                tf.config.optimizer.set_jit(True)
                logger.info("XLA compilation enabled for TensorFlow model")
            except:
                logger.warning("Failed to enable XLA compilation")
        
        # Configure device placement
        if self.config.device != "auto":
            if self.config.device == "cpu":
                # Disable GPU
                try:
                    tf.config.set_visible_devices([], 'GPU')
                    logger.info("TensorFlow configured to use CPU only")
                except:
                    logger.warning("Failed to disable GPU, TensorFlow will determine device placement")
            elif self.config.device.startswith("cuda") or self.config.device.startswith("gpu"):
                # Use specific GPU if specified (e.g. "cuda:0" or "gpu:1")
                if ":" in self.config.device:
                    gpu_id = int(self.config.device.split(":")[-1])
                    try:
                        gpus = tf.config.list_physical_devices('GPU')
                        if gpu_id < len(gpus):
                            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                            logger.info(f"TensorFlow configured to use GPU {gpu_id}")
                        else:
                            logger.warning(f"GPU {gpu_id} not found. Using default device")
                    except:
                        logger.warning("Failed to set specific GPU, TensorFlow will determine device placement")
            elif self.config.device.startswith("tpu"):
                # Use TPU if specified
                if self.config.tpu_name:
                    try:
                        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=self.config.tpu_name)
                        tf.config.experimental_connect_to_cluster(resolver)
                        tf.tpu.experimental.initialize_tpu_system(resolver)
                        logger.info(f"TPU {self.config.tpu_name} initialized")
                    except:
                        logger.warning(f"Failed to initialize TPU {self.config.tpu_name}")
        
        # Set up distribution strategy if configured
        if self.config.distribute_strategy:
            try:
                if self.config.distribute_strategy == "mirrored":
                    self._distribution_strategy = tf.distribute.MirroredStrategy()
                    logger.info("Using MirroredStrategy for distribution")
                elif self.config.distribute_strategy == "tpu" and self.config.tpu_name:
                    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=self.config.tpu_name)
                    tf.config.experimental_connect_to_cluster(resolver)
                    tf.tpu.experimental.initialize_tpu_system(resolver)
                    self._distribution_strategy = tf.distribute.TPUStrategy(resolver)
                    logger.info(f"Using TPUStrategy with TPU {self.config.tpu_name}")
                elif self.config.distribute_strategy == "parameter_server":
                    self._distribution_strategy = tf.distribute.experimental.ParameterServerStrategy()
                    logger.info("Using ParameterServerStrategy for distribution")
                elif self.config.distribute_strategy == "multi_worker_mirrored":
                    self._distribution_strategy = tf.distribute.MultiWorkerMirroredStrategy()
                    logger.info("Using MultiWorkerMirroredStrategy for distribution")
                else:
                    logger.warning(f"Unknown distribution strategy: {self.config.distribute_strategy}")
            except Exception as e:
                logger.warning(f"Failed to initialize distribution strategy: {str(e)}")
                self._distribution_strategy = None
        
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load a TensorFlow model from disk.
        
        Args:
            model_path: Path to the TensorFlow model file or directory
            
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is not installed. Cannot load TensorFlow model.")
            return False
        
        if model_path is None:
            logger.error("Model path is required to load a TensorFlow model")
            return False
        
        try:
            start_time = time.time()
            
            # Determine model format
            if os.path.isdir(model_path):
                # Directory path is likely a SavedModel
                logger.info(f"Loading TensorFlow SavedModel from {model_path}")
                
                # Use distribution strategy if configured
                if self._distribution_strategy:
                    with self._distribution_strategy.scope():
                        self._model = tf.keras.models.load_model(model_path, compile=False)
                else:
                    self._model = tf.keras.models.load_model(model_path, compile=False)
                
                # Get signatures if available
                try:
                    self._signatures = tf.saved_model.load(model_path).signatures
                    logger.debug(f"Loaded model signatures: {list(self._signatures.keys())}")
                except:
                    logger.debug("No signatures found in SavedModel")
                    
            elif model_path.endswith(('.h5', '.keras')):
                # H5 or new Keras format
                logger.info(f"Loading TensorFlow H5/Keras model from {model_path}")
                
                # Use distribution strategy if configured
                if self._distribution_strategy:
                    with self._distribution_strategy.scope():
                        self._model = tf.keras.models.load_model(model_path, compile=False)
                else:
                    self._model = tf.keras.models.load_model(model_path, compile=False)
                    
            elif model_path.endswith('.tflite'):
                # TFLite model
                logger.info(f"Loading TensorFlow Lite model from {model_path}")
                self._model = tf.lite.Interpreter(model_path=model_path)
                self._model.allocate_tensors()
                
            else:
                logger.error(f"Unsupported TensorFlow model format for {model_path}")
                return False
            
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
            logger.info(f"TensorFlow model '{self.name}' loaded in {load_time:.2f}s")
            
            # Log model summary if it's a Keras model
            if hasattr(self._model, 'summary') and callable(self._model.summary):
                if logger.isEnabledFor(logging.DEBUG):
                    string_io = tf.io.StringIO()
                    self._model.summary(print_fn=lambda x: string_io.write(x + '\n'))
                    logger.debug(f"Model summary:\n{string_io.getvalue()}")
                    
            return True
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {str(e)}")
            return False
    
    def predict(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference on the TensorFlow model.
        
        Args:
            inputs: Input data for model inference
            **kwargs: Additional inference parameters
            
        Returns:
            Model predictions
        """
        if not self.is_loaded or self._model is None:
            raise ValueError("Model is not loaded. Call load() before predict().")
        
        try:
            start_time = time.time()
            
            # Check if we're using a TFLite model
            if hasattr(self._model, 'get_input_details'):
                # TFLite model
                input_details = self._model.get_input_details()
                output_details = self._model.get_output_details()
                
                # Convert inputs to NumPy array if needed
                if not isinstance(inputs, np.ndarray):
                    inputs = np.array(inputs, dtype=np.float32)
                
                # Resize input tensor if dynamic shape model
                if self.config.dynamic_shape and input_details[0]['shape_signature'][0] == -1:
                    input_shape = list(input_details[0]['shape'])
                    input_shape[0] = inputs.shape[0]
                    self._model.resize_tensor_input(input_details[0]['index'], input_shape)
                    self._model.allocate_tensors()
                
                # Set input tensor
                self._model.set_tensor(input_details[0]['index'], inputs)
                
                # Run inference
                self._model.invoke()
                
                # Get output tensor
                result = self._model.get_tensor(output_details[0]['index'])
            else:
                # Regular Keras model
                # Check if signature is provided and the model has signatures
                if 'signature' in kwargs and self._signatures:
                    signature_name = kwargs.pop('signature')
                    if signature_name in self._signatures:
                        # Use the specified signature
                        if isinstance(inputs, np.ndarray):
                            inputs = tf.convert_to_tensor(inputs)
                        
                        # Prepare inputs for the signature
                        if isinstance(inputs, dict):
                            result = self._signatures[signature_name](**inputs)
                        else:
                            # Assume the signature expects a single input
                            input_name = list(self._signatures[signature_name].structured_input_signature[1].keys())[0]
                            result = self._signatures[signature_name](**{input_name: inputs})
                        
                        # Convert result to numpy if it's a TensorFlow tensor
                        if isinstance(result, dict):
                            result = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in result.items()}
                        elif hasattr(result, 'numpy'):
                            result = result.numpy()
                    else:
                        logger.warning(f"Signature '{signature_name}' not found. Using default predict method.")
                        result = self._model.predict(inputs, **kwargs)
                else:
                    # Use the regular predict method
                    batch_size = kwargs.pop('batch_size', self.

"""
TensorFlow model implementation for AutoAI-AgentRAG.

This module provides a TensorFlow-specific implementation of the MLModel class
for integrating TensorFlow models into the AutoAI-AgentRAG framework.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from autoai_agentrag.ml.model import MLModel, ModelConfig, ModelFramework, ModelMetadata

logger = logging.getLogger(__name__)

# Try to import TensorFlow, but don't fail if it's not installed
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not installed. TensorFlowModel will not be functional.")
    TENSORFLOW_AVAILABLE = False
    # Create a placeholder for type hints
    class tf:
        class keras:
            class Model:
                pass
            class Sequential:
                pass

class TFModelFormat(str):
    """Supported TensorFlow model formats."""
    
    SAVED_MODEL = "saved_model"
    H5 = "h5"
    KERAS = "keras"
    TF_LITE = "tflite"


class TensorFlowConfig(ModelConfig):
    """TensorFlow-specific configuration parameters."""
    
    eager_mode: bool = True
    mixed_precision: bool = False
    xla_compilation: bool = False
    model_format: TFModelFormat = TFModelFormat.SAVED_MODEL
    dynamic_shape: bool = True
    use_tensorrt: bool = False


class TensorFlowModel(MLModel):
    """
    TensorFlow-specific implementation of the MLModel class.
    
    This class provides TensorFlow-specific methods for loading, saving, and
    running inference with TensorFlow models.
    
    Attributes:
        name (str): Name of the model
        metadata (ModelMetadata): Metadata for this model
        config (TensorFlowConfig): TensorFlow-specific configuration
        is_loaded (bool): Whether the model is loaded and ready for inference
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[TensorFlowConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new TensorFlow model.
        
        Args:
            name: Name of the model
            config: Optional TensorFlow-specific configuration
            metadata: Optional metadata for the model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Please install TensorFlow to use TensorFlowModel.")
        
        super().__init__(
            name=name,
            framework=ModelFramework.TENSORFLOW,
            config=config or TensorFlowConfig(),
            metadata=metadata
        )
        
        # Cast config to TensorFlowConfig for type hints
        self.config = cast(TensorFlowConfig, self.config)
        
        # Configure TensorFlow based on the config
        self._configure_tensorflow()
        
        # Initialize model attribute
        self._model: Optional[Union[tf.keras.Model, tf.keras.Sequential]] = None
        self._signatures = {}
        
    def _configure_tensorflow(self):
        """Configure TensorFlow based on the model configuration."""
        if not TENSORFLOW_AVAILABLE:
            return
        
        # Set eager execution mode
        tf.config.run_functions_eagerly(self.config.eager_mode)
        
        # Configure mixed precision if requested
        if self.config.mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled for TensorFlow model")
        
        # Configure XLA compilation
        if self.config.xla_compilation:
            tf.config.optimizer.set_jit(True)
            logger.info("XLA compilation enabled for TensorFlow model")
        
        # Configure device placement
        if self.config.device != "auto":
            if self.config.device == "cpu":
                # Disable GPU
                try:
                    tf.config.set_visible_devices([], 'GPU')
                    logger.info("TensorFlow configured to use CPU only")
                except:
                    logger.warning("Failed to disable GPU, TensorFlow will determine device placement")
            elif self.config.device.startswith("cuda") or self.config.device.startswith("gpu"):
                # Use specific GPU if specified (e.g. "cuda:0" or "gpu:1")
                if ":" in self.config.device:
                    gpu_id = int(self.config.device.split(":")[-1])
                    try:
                        gpus = tf.config.list_physical_devices('GPU')
                        if gpu_id < len(gpus):
                            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                            logger.info(f"TensorFlow configured to use GPU {gpu_id}")
                        else:
                            logger.warning(f"GPU {gpu_id} not found. Using default device")
                    except:
                        logger.warning("Failed to set specific GPU, TensorFlow will determine device placement")
        
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load a TensorFlow model from disk.
        
        Args:
            model_path: Path to the TensorFlow model file or directory
            
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow is not installed. Cannot load TensorFlow model.")
            return False
        
        if model_path is None:
            logger.error("Model path is required to load a TensorFlow model")
            return False
        
        try:
            start_time = time.time()
            
            # Determine model format
            if os.path.isdir(model_path):
                # Directory path is likely a SavedModel
                logger.info(f"Loading TensorFlow SavedModel from {model_path}")
                self._model = tf.keras.models.load_model(model_path)
                # Get signatures if available
                if hasattr(self._model, 'signatures'):
                    self._signatures = self._model.signatures
            elif model_path.endswith(('.h5', '.keras')):
                # H5 or new Keras format
                logger.info(f"Loading TensorFlow H5/Keras model from {model_path}")
                self._model = tf.keras.models.load_model(model_path)
            elif model_path.endswith('.tflite'):
                # TFLite model
                logger.info(f"Loading TensorFlow Lite model from {model_path}")
                self._model = tf.lite.Interpreter(model_path=model_path)
                self._model.allocate_tensors()
            else:
                logger.error(f"Unsupported TensorFlow model format for {model_path}")
                return False
            
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
            logger.info(f"TensorFlow model '{self.name}' loaded in {load_time:.2f}s")
            
            # Log model summary if it's a Keras model
            if hasattr(self._model, 'summary') and callable(self._model.summary):
                if logger.isEnabledFor(logging.DEBUG):
                    self._model.summary(print_fn=lambda x: logger.debug(x))
                    
            return True
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {str(e)}")
            return False
    
    def predict(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference on the TensorFlow model.
        
        Args:
            inputs: Input data for model inference
            **kwargs: Additional inference parameters
            
        Returns:
            Model predictions
        """
        if not self.is_loaded or self._model is None:
            raise ValueError("Model is not loaded. Call load() before predict().")
        
        try:
            start_time = time.time()
            
            # Check if we're using a TFLite model
            if hasattr(self._model, 'get_input_details'):
                # TFLite model
                input_details = self._model.get_input_details()
                output_details = self._model.get_output_details()
                
                # Convert inputs to NumPy array if needed
                if not isinstance(inputs, np.ndarray):
                    inputs = np.array(inputs, dtype=input_details[0]['dtype'])
                
                # Set input tensor
                self._model.set_tensor(input_details[0]['index'], inputs)
                
                # Run inference
                self._model.invoke()
                
                # Get output tensor
                result = self._model.get_tensor(output_details[0]['index'])
            else:
                # Regular Keras model
                # Check if signature is provided and the model has signatures
                if 'signature' in kwargs and hasattr(self._model, 'signatures'):
                    signature_name = kwargs.pop('signature')
                    if signature_name in self._signatures:
                        # Use the specified signature
                        result = self._signatures[signature_name](**inputs)
                    else:
                        logger.warning(f"Signature '{signature_name}' not found. Using default predict method.")
                        result = self._model.predict(inputs, **kwargs)
                else:
                    # Use the regular predict method
                    result = self._model.predict(inputs, **kwargs)
            
            # Calculate and log inference time
            inference_time = time.time() - start_time
            logger.debug(f"TensorFlow inference time: {inference_time:.4f}s")
            
            return result
        except Exception as e:
            logger.error(f"Error during TensorFlow model inference: {str(e)}")
            raise
    
    def save(self, save_path: str) -> bool:
        """
        Save the TensorFlow model to disk.
        
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
            
            # Determine save format based on the path extension or config
            if hasattr(self._model, 'save'):  # Only Keras models have save method
                if save_path.endswith('.h5'):
                    self._model.save(save_path, save_format='h5')
                elif save_path.endswith('.keras'):
                    self._model.save(save_path, save_format='keras')
                elif os.path.isdir(save_path) or not os.path.splitext(save_path)[1]:
                    # If it's a directory or has no extension, save as SavedModel
                    self._model.save(save_path, save_format='tf')
                else:
                    # Default to SavedModel
                    self._model.save(save_path)
                
                # Save metadata
                meta_path = f"{save_path}.meta.json"
                self.save_metadata(meta_path)
                
                logger.info(f"TensorFlow model saved to {save_path}")
                return True
            elif hasattr(self._model, 'get_input_details'):  # TFLite model
                logger.warning("Saving TFLite models is not supported. The model must be converted from a regular TensorFlow model.")
                return False
            else:
                logger.error("Unknown model type, cannot save")
                return False
        except Exception as e:
            logger.error(f"Error saving TensorFlow model: {str(e)}")
            return False
    
    def convert_to_tflite(self, save_path: str, optimization: str = "DEFAULT") -> bool:
        """
        Convert the model to TensorFlow Lite format.
        
        Args:
            save_path: Path to save the TFLite model to
            optimization: Optimization level (DEFAULT, OPTIMIZE_FOR_SIZE, OPTIMIZE_FOR_LATENCY)
            
        Returns:
            True if the model was converted and saved successfully, False otherwise
        """
        if not self.is_loaded or self._model is None:
            logger.warning("Cannot convert model that isn't loaded")
            return False
        
        try:
            # Only Keras models can be converted to TFLite
            if not hasattr(self._model, 'save'):
                logger.error("Only Keras models can be converted to TFLite")
                return False
            
            # Import TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(self._model)
            
            # Set optimization level
            if optimization == "OPTIMIZE_FOR_SIZE":
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            elif optimization == "OPTIMIZE_FOR_LATENCY":
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
            elif optimization != "DEFAULT":
                logger.warning(f"Unknown optimization level: {optimization}. Using DEFAULT.")
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save model
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Model converted to TFLite and saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error converting to TFLite: {str(e)}")
            return False

