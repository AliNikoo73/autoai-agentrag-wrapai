"""
Unit tests for the TensorFlow model implementation in AutoAI-AgentRAG.

This module contains tests for the TensorFlowModel class and related utilities.
"""

import os
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np

# Mock the TensorFlow module before importing the TensorFlowModel
tensorflow_mock = MagicMock()
keras_mock = MagicMock()
tensorflow_mock.keras = keras_mock
tensorflow_mock.keras.mixed_precision = MagicMock()
tensorflow_mock.keras.mixed_precision.Policy = MagicMock()
tensorflow_mock.keras.mixed_precision.set_global_policy = MagicMock()
tensorflow_mock.config = MagicMock()
tensorflow_mock.config.run_functions_eagerly = MagicMock()
tensorflow_mock.config.optimizer = MagicMock()
tensorflow_mock.config.list_physical_devices = MagicMock(return_value=['gpu0', 'gpu1'])
tensorflow_mock.config.set_visible_devices = MagicMock()
tensorflow_mock.config.experimental = MagicMock()
tensorflow_mock.lite = MagicMock()
tensorflow_mock.lite.TFLiteConverter = MagicMock()
tensorflow_mock.lite.Interpreter = MagicMock()
tensorflow_mock.distribute = MagicMock()
tensorflow_mock.saved_model = MagicMock()
tensorflow_mock.__version__ = "2.8.0"

# Add the mock to sys.modules
with patch.dict('sys.modules', {'tensorflow': tensorflow_mock}):
    # Now import the TensorFlowModel
    from autoai_agentrag.ml.tensorflow import (
        TensorFlowModel, TensorFlowConfig, TFModelFormat
    )
    from autoai_agentrag.ml.model import ModelMetadata, ModelFramework


class TestTensorFlowConfig(unittest.TestCase):
    """Test cases for the TensorFlowConfig class."""
    
    def test_config_defaults(self):
        """Test that TensorFlowConfig has correct default values."""
        config = TensorFlowConfig()
        
        self.assertTrue(config.eager_mode)
        self.assertFalse(config.mixed_precision)
        self.assertFalse(config.xla_compilation)
        self.assertEqual(config.model_format, TFModelFormat.SAVED_MODEL)
        self.assertTrue(config.dynamic_shape)
        self.assertFalse(config.use_tensorrt)
        self.assertIsNone(config.tpu_name)
        self.assertIsNone(config.distribute_strategy)
        self.assertTrue(config.allow_soft_placement)
        self.assertFalse(config.enable_op_determinism)
        self.assertTrue(config.memory_growth)
        self.assertTrue(config.use_graph_optimization)
        
    def test_config_custom_values(self):
        """Test that TensorFlowConfig accepts custom values."""
        config = TensorFlowConfig(
            eager_mode=False,
            mixed_precision=True,
            xla_compilation=True,
            model_format=TFModelFormat.H5,
            dynamic_shape=False,
            use_tensorrt=True,
            tpu_name="tpu-v3",
            distribute_strategy="mirrored",
            device="gpu:0",
            batch_size=64
        )
        
        self.assertFalse(config.eager_mode)
        self.assertTrue(config.mixed_precision)
        self.assertTrue(config.xla_compilation)
        self.assertEqual(config.model_format, TFModelFormat.H5)
        self.assertFalse(config.dynamic_shape)
        self.assertTrue(config.use_tensorrt)
        self.assertEqual(config.tpu_name, "tpu-v3")
        self.assertEqual(config.distribute_strategy, "mirrored")
        self.assertEqual(config.device, "gpu:0")
        self.assertEqual(config.batch_size, 64)


@patch.dict('sys.modules', {'tensorflow': tensorflow_mock})
class TestTensorFlowModel(unittest.TestCase):
    """Test cases for the TensorFlowModel class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Reset all mocks
        tensorflow_mock.reset_mock()
        keras_mock.reset_mock()
        
        # Create a model instance
        self.model_name = "tf-test-model"
        self.model = TensorFlowModel(name=self.model_name)
        
    def test_model_initialization(self):
        """Test that TensorFlowModel initializes with correct values."""
        self.assertEqual(self.model.name, self.model_name)
        self.assertEqual(self.model.metadata.framework, ModelFramework.TENSORFLOW)
        self.assertIsInstance(self.model.config, TensorFlowConfig)
        self.assertFalse(self.model.is_loaded)
        self.assertIsNone(self.model._model)
        self.assertEqual(self.model._signatures, {})
        self.assertIsNone(self.model._distribution_strategy)
        
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = TensorFlowConfig(
            eager_mode=False,
            mixed_precision=True,
            xla_compilation=True
        )
        model = TensorFlowModel(name="custom-config", config=config)
        
        self.assertEqual(model.config.eager_mode, False)
        self.assertEqual(model.config.mixed_precision, True)
        self.assertEqual(model.config.xla_compilation, True)
        
        # Verify TensorFlow was configured correctly
        tensorflow_mock.config.run_functions_eagerly.assert_called_with(False)
        
    def test_configure_tensorflow(self):
        """Test the _configure_tensorflow method."""
        # Reset mock
        tensorflow_mock.reset_mock()
        
        # Create model with specific config for testing
        config = TensorFlowConfig(
            eager_mode=False,
            mixed_precision=True,
            xla_compilation=True,
            enable_op_determinism=True,
            memory_growth=True,
            device="cpu"
        )
        model = TensorFlowModel(name="config-test", config=config)
        
        # Verify configuration actions
        tensorflow_mock.config.run_functions_eagerly.assert_called_with(False)
        tensorflow_mock.keras.mixed_precision.Policy.assert_called_with('mixed_float16')
        tensorflow_mock.keras.mixed_precision.set_global_policy.assert_called_once()
        tensorflow_mock.config.optimizer.set_jit.assert_called_with(True)
        tensorflow_mock.config.experimental.enable_op_determinism.assert_called_once()
        tensorflow_mock.config.list_physical_devices.assert_called_with('GPU')
        tensorflow_mock.config.set_visible_devices.assert_called_once()
        
    def test_configure_gpu_devices(self):
        """Test configuration with specific GPU devices."""
        tensorflow_mock.reset_mock()
        tensorflow_mock.config.list_physical_devices.return_value = ['gpu0', 'gpu1', 'gpu2']
        
        config = TensorFlowConfig(device="gpu:1")
        model = TensorFlowModel(name="gpu-test", config=config)
        
        tensorflow_mock.config.list_physical_devices.assert_called_with('GPU')
        tensorflow_mock.config.set_visible_devices.assert_called_with('gpu1', 'GPU')
        
    def test_distribution_strategy_setup(self):
        """Test setting up distribution strategies."""
        tensorflow_mock.reset_mock()
        
        # Mock strategy objects
        mirrored_strategy_mock = MagicMock()
        tensorflow_mock.distribute.MirroredStrategy.return_value = mirrored_strategy_mock
        
        # Create model with mirrored strategy
        config = TensorFlowConfig(distribute_strategy="mirrored")
        model = TensorFlowModel(name="strategy-test", config=config)
        
        # Verify strategy was created
        tensorflow_mock.distribute.MirroredStrategy.assert_called_once()
        self.assertEqual(model._distribution_strategy, mirrored_strategy_mock)
        
    @patch('os.path.isdir')
    def test_load_saved_model(self, mock_isdir):
        """Test loading a SavedModel."""
        # Setup
        mock_isdir.return_value = True
        model_path = "/path/to/saved_model"
        mock_keras_model = MagicMock()
        tensorflow_mock.keras.models.load_model.return_value = mock_keras_model
        mock_signatures = {"serving_default": MagicMock()}
        tensorflow_mock.saved_model.load.return_value.signatures = mock_signatures
        
        # Execute
        result = self.model.load(model_path)
        
        # Assert
        self.assertTrue(result)
        self.assertTrue(self.model.is_loaded)
        self.assertEqual(self.model._model, mock_keras_model)
        self.assertEqual(self.model._signatures, mock_signatures)
        tensorflow_mock.keras.models.load_model.assert_called_with(model_path, compile=False)
        
    def test_load_h5_model(self):
        """Test loading an H5 model."""
        # Setup
        model_path = "/path/to/model.h5"
        mock_keras_model = MagicMock()
        tensorflow_mock.keras.models.load_model.return_value = mock_keras_model
        
        # Execute
        result = self.model.load(model_path)
        
        # Assert
        self.assertTrue(result)
        self.assertTrue(self.model.is_loaded)
        self.assertEqual(self.model._model, mock_keras_model)
        tensorflow_mock.keras.models.load_model.assert_called_with(model_path, compile=False)
        
    def test_load_tflite_model(self):
        """Test loading a TFLite model."""
        # Setup
        model_path = "/path/to/model.tflite"
        mock_interpreter = MagicMock()
        tensorflow_mock.lite.Interpreter.return_value = mock_interpreter
        
        # Execute
        result = self.model.load(model_path)
        
        # Assert
        self.assertTrue(result)
        self.assertTrue(self.model.is_loaded)
        self.assertEqual(self.model._model, mock_interpreter)
        tensorflow_mock.lite.Interpreter.assert_called_with(model_path=model_path)
        mock_interpreter.allocate_tensors.assert_called_once()
        
    def test_load_with_distribution_strategy(self):
        """Test loading a model with distribution strategy."""
        # Setup
        model_path = "/path/to/model.h5"
        mock_keras_model = MagicMock()
        tensorflow_mock.keras.models.load_model.return_value = mock_keras_model
        
        # Create a mock distribution strategy
        mock_strategy = MagicMock()
        self.model._distribution_strategy = mock_strategy
        
        # Execute
        result = self.model.load(model_path)
        
        # Assert
        self.assertTrue(result)
        self.assertTrue(self.model.is_loaded)
        mock_strategy.__enter__.assert_called()  # Check that context manager was used
        
    def test_load_error_handling(self):
        """Test error handling when loading a model."""
        # Setup
        model_path = "/path/to/model.h5"
        tensorflow_mock.keras.models.load_model.side_effect = Exception("Load error")
        
        # Execute
        result = self.model.load(model_path)
        
        # Assert
        self.assertFalse(result)
        self.assertFalse(self.model.is_loaded)
        
    def test_predict_keras_model(self):
        """Test prediction with a Keras model."""
        # Setup
        mock_keras_model = MagicMock()
        mock_keras_model.predict.return_value = np.array([1, 2, 3])
        
        # Set model and mark as loaded
        self.model._model = mock_keras_model
        self.model.is_loaded = True
        
        # Execute
        inputs = np.array([[1, 2, 3]])
        result = self.model.predict(inputs, batch_size=1)
        
        # Assert
        self.assertTrue(np.array_equal(result, np.array([1, 2, 3])))
        mock_keras_model.predict.assert_called_with(inputs, batch_size=1)
        
    def test_predict_tflite_model(self):
        """Test prediction with a TFLite model."""
        # Setup
        mock_interpreter = MagicMock()
        mock_interpreter.get_input_details.return_value = [{'index': 0, 'shape': [1, 3], 'dtype': np.float32}]
        mock_interpreter.get_output_details.return_value = [{'index': 0}]
        mock_interpreter.get_tensor.return_value = np.array([4, 5, 6])
        
        # Set model and mark as loaded
        self.model._model = mock_interpreter
        self.model.is_loaded = True
        
        # Execute
        inputs = np.array([[1, 2, 3]])
        result = self.model.predict(inputs)
        
        # Assert
        self.assertTrue(np.array_equal(result, np.array([4, 5, 6])))
        mock_interpreter.set_tensor.assert_called_with(0, inputs)
        mock_interpreter.invoke.assert_called_once()
        mock_interpreter.get_tensor.assert_called_once()
        
    def test_predict_with_dynamic_shapes(self):
        """Test prediction with dynamic shapes in TFLite."""
        # Setup
        mock_interpreter = MagicMock()
        mock_interpreter.get_input_details.return_value = [
            {'index': 0, 'shape': [1, 3], 'shape_signature': [-1, 3], 'dtype': np.float32}
        ]
        mock_interpreter.get_output_details.return_value = [{'index': 0}]
        mock_interpreter.get_tensor.return_value = np.array([4, 5, 6])
        
        # Set model and mark as loaded
        self.model._model = mock_interpreter
        self.model.is_loaded = True
        self.model.config.dynamic_shape = True
        
        # Execute
        inputs = np.array([[1, 2, 3]])
        result = self.model.predict(inputs)
        
        # Assert
        mock_interpreter.resize_tensor_input.assert_called_with(0, [1, 3])
        mock_interpreter.allocate_tensors.assert_called_once()
        
    def test_predict_with_signature(self):
        """Test prediction using a specific model signature."""
        # Setup
        mock_keras_model = MagicMock()
        mock_signature = MagicMock()
        mock_signature.return_value = {'output': MagicMock(numpy=MagicMock(return_value=np.array([7, 8, 9])))}
        mock_signature.structured_input_signature = ((), {'input_

