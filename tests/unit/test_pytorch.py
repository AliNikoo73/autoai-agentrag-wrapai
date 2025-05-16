"""
Unit tests for the PyTorch model implementation in AutoAI-AgentRAG.

This module contains tests for the PyTorchModel class and related utilities.
"""

import os
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np

# Mock the PyTorch module before importing the PyTorchModel
torch_mock = MagicMock()
torch_mock.nn = MagicMock()
torch_mock.nn.Module = MagicMock
torch_mock.cuda = MagicMock()
torch_mock.cuda.is_available = MagicMock(return_value=True)
torch_mock.device = MagicMock()
torch_mock.device.side_effect = lambda x: x  # Return the device string as is
torch_mock.rand = MagicMock(return_value=MagicMock())
torch_mock.tensor = MagicMock(return_value=MagicMock())
torch_mock.jit = MagicMock()
torch_mock.jit.ScriptModule = MagicMock
torch_mock.jit.trace = MagicMock()
torch_mock.jit.load = MagicMock()
torch_mock.load = MagicMock()
torch_mock.save = MagicMock()
torch_mock.onnx = MagicMock()
torch_mock.backends = MagicMock()
torch_mock.backends.cudnn = MagicMock()
torch_mock.__version__ = "1.13.0"

# Add mock attributes to CPU and CUDA tensors
tensor_mock = MagicMock()
tensor_mock.device = "cuda:0"
tensor_mock.to = MagicMock(return_value=tensor_mock)
tensor_mock.detach = MagicMock(return_value=tensor_mock)
tensor_mock.cpu = MagicMock(return_value=tensor_mock)
tensor_mock.numpy = MagicMock(return_value=np.array([1, 2, 3]))

torch_mock.tensor.return_value = tensor_mock

# Add the mock to sys.modules
with patch.dict('sys.modules', {'torch': torch_mock}):
    # Now import the PyTorchModel
    from autoai_agentrag.ml.pytorch import (
        PyTorchModel, PyTorchConfig
    )
    from autoai_agentrag.ml.model import ModelMetadata, ModelFramework


class TestPyTorchConfig(unittest.TestCase):
    """Test cases for the PyTorchConfig class."""
    
    def test_config_defaults(self):
        """Test that PyTorchConfig has correct default values."""
        config = PyTorchConfig()
        
        self.assertFalse(config.use_jit)
        self.assertFalse(config.use_fp16)
        self.assertTrue(config.use_cuda)
        self.assertTrue(config.cudnn_benchmark)
        self.assertFalse(config.cudnn_deterministic)
        self.assertEqual(config.model_format, "pt")
        self.assertFalse(config.trace_model)
        self.assertEqual(config.num_workers, 4)
        self.assertTrue(config.pin_memory)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.batch_size, 32)
        
    def test_config_custom_values(self):
        """Test that PyTorchConfig accepts custom values."""
        config = PyTorchConfig(
            use_jit=True,
            use_fp16=True,
            use_cuda=False,
            cudnn_benchmark=False,
            cudnn_deterministic=True,
            model_format="script",
            trace_model=True,
            num_workers=2,
            pin_memory=False,
            device="cpu",
            batch_size=64
        )
        
        self.assertTrue(config.use_jit)
        self.assertTrue(config.use_fp16)
        self.assertFalse(config.use_cuda)
        self.assertFalse(config.cudnn_benchmark)
        self.assertTrue(config.cudnn_deterministic)
        self.assertEqual(config.model_format, "script")
        self.assertTrue(config.trace_model)
        self.assertEqual(config.num_workers, 2)
        self.assertFalse(config.pin_memory)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.batch_size, 64)


@patch.dict('sys.modules', {'torch': torch_mock})
class TestPyTorchModel(unittest.TestCase):
    """Test cases for the PyTorchModel class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Reset all mocks
        torch_mock.reset_mock()
        
        # Create a model instance
        self.model_name = "pytorch-test-model"
        self.model = PyTorchModel(name=self.model_name)
        
    def test_model_initialization(self):
        """Test that PyTorchModel initializes with correct values."""
        self.assertEqual(self.model.name, self.model_name)
        self.assertEqual(self.model.metadata.framework, ModelFramework.PYTORCH)
        self.assertIsInstance(self.model.config, PyTorchConfig)
        self.assertFalse(self.model.is_loaded)
        self.assertIsNone(self.model._model)
        
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = PyTorchConfig(
            use_jit=True,
            use_fp16=True,
            use_cuda=False,
            device="cpu"
        )
        model = PyTorchModel(name="custom-config", config=config)
        
        self.assertEqual(model.config.use_jit, True)
        self.assertEqual(model.config.use_fp16, True)
        self.assertEqual(model.config.use_cuda, False)
        self.assertEqual(model.config.device, "cpu")
        
        # Verify PyTorch was configured correctly
        torch_mock.device.assert_called_with("cpu")
        
    def test_configure_pytorch(self):
        """Test the _configure_pytorch method."""
        # Reset mock
        torch_mock.reset_mock()
        
        # Create model with specific config for testing
        config = PyTorchConfig(
            cudnn_benchmark=True,
            cudnn_deterministic=False,
            device="cuda:0",
            use_cuda=True
        )
        model = PyTorchModel(name="config-test", config=config)
        
        # Verify configuration actions
        torch_mock.backends.cudnn.benchmark = True
        torch_mock.backends.cudnn.deterministic = False
        torch_mock.device.assert_called_with("cuda:0")
        
    def test_configure_cpu_device(self):
        """Test configuration with CPU device."""
        torch_mock.reset_mock()
        
        config = PyTorchConfig(device="cpu", use_cuda=False)
        model = PyTorchModel(name="cpu-test", config=config)
        
        torch_mock.device.assert_called_with("cpu")
        
    def test_configure_cuda_device_not_available(self):
        """Test fallback to CPU when CUDA requested but not available."""
        torch_mock.reset_mock()
        torch_mock.cuda.is_available.return_value = False
        
        config = PyTorchConfig(device="cuda:0")
        model = PyTorchModel(name="gpu-test", config=config)
        
        # Should fall back to CPU
        torch_mock.device.assert_called_with("cpu")
        
    def test_configure_auto_device(self):
        """Test auto device selection."""
        torch_mock.reset_mock()
        
        # Test with CUDA available
        torch_mock.cuda.is_available.return_value = True
        config = PyTorchConfig(device="auto")
        model = PyTorchModel(name="auto-test", config=config)
        torch_mock.device.assert_called_with("cuda:0")
        
        # Test with CUDA not available
        torch_mock.reset_mock()
        torch_mock.cuda.is_available.return_value = False
        config = PyTorchConfig(device="auto")
        model = PyTorchModel(name="auto-test", config=config)
        torch_mock.device.assert_called_with("cpu")
        
    def test_load_pt_model_state_dict(self):
        """Test loading a PyTorch model from state dict."""
        # Setup
        model_path = "/path/to/model.pt"
        mock_state_dict = {"layer1.weight": torch_mock.tensor}
        torch_mock.load.return_value = mock_state_dict
        
        # Create a method for model creation
        self.model.create_model = MagicMock()
        mock_model = MagicMock()
        self.model.create_model.return_value = mock_model
        
        # Execute
        with patch("os.path.exists", return_value=True):
            result = self.model.load(model_path)
        
        # Assert
        self.assertTrue(result)
        self.assertTrue(self.model.is_loaded)
        self.assertEqual(self.model._model, mock_model)
        torch_mock.load.assert_called_with(model_path, map_location=self.model._device)
        mock_model.load_state_dict.assert_called_with(mock_state_dict)
        mock_model.to.assert_called_with(self.model._device)
        mock_model.eval.assert_called_once()
        
    def test_load_jit_model(self):
        """Test loading a TorchScript (JIT) model."""
        # Setup
        model_path = "/path/to/model.pt"
        mock_jit_model = MagicMock()
        torch_mock.jit.load.return_value = mock_jit_model
        
        # Configure to use JIT
        self.model.config.use_jit = True
        
        # Execute
        with patch("os.path.exists", return_value=True):
            result = self.model.load(model_path)
        
        # Assert
        self.assertTrue(result)
        self.assertTrue(self.model.is_loaded)
        self.assertEqual(self.model._model, mock_jit_model)
        torch_mock.jit.load.assert_called_with(model_path, map_location=self.model._device)
        
    def test_load_checkpoint_with_state_dict(self):
        """Test loading a checkpoint with state_dict and metadata."""
        # Setup
        model_path = "/path/to/model.pt"
        mock_checkpoint = {
            "state_dict": {"layer1.weight": torch_mock.tensor},
            "metadata": {
                "name": "checkpoint-model",
                "framework": "pytorch",
                "version": "1.0.0"
            }
        }
        torch_mock.load.return_value = mock_checkpoint
        
        # Create a method for model creation
        self.model.create_model = MagicMock()
        mock_model = MagicMock()
        self.model.create_model.return_value = mock_model
        
        # Execute
        with patch("os.path.exists", return_value=True):
            result = self.model.load(model_path)
        
        # Assert
        self.assertTrue(result)
        self.assertTrue(self.model.is_loaded)
        self.assertEqual(self.model._model, mock_model)
        self.assertEqual(self.model.metadata.name, "checkpoint-model")
        self.assertEqual(self.model.metadata.version, "1.0.0")
        mock_model.load_state_dict.assert_called_with(mock_checkpoint["state_dict"])
        
    def test_load_full_model(self):
        """Test loading a full model (not state dict or checkpoint)."""
        # Setup
        model_path = "/path/to/model.pt"
        mock_model = MagicMock()
        torch_mock.load.return_value = mock_model
        
        # We need to make the isinstance check work with our mock
        torch_mock.nn.Module.__instancecheck__ = MagicMock(return_value=True)
        
        # Execute
        with patch("os.path.exists", return_value=True):
            result = self.model.load(model_path)
        
        # Assert
        self.assertTrue(result)
        self.assertTrue(self.model.is_loaded)
        self.assertEqual(self.model._model, mock_model)
        mock_model.to.assert_called_with(self.model._device)
        mock_model.eval.assert_called_once()
        
    def test_load_error_handling(self):
        """Test error handling when loading a model."""
        # Setup
        model_path = "/path/to/model.pt"
        torch_mock.load.side_effect = Exception("Load error")
        
        # Execute
        with patch("os.path.exists", return_value=True):
            result = self.model.load(model_path)
        
        # Assert
        self.assertFalse(result)
        self.assertFalse(self.model.is_loaded)
        
    def test_load_convert_to_jit(self):
        """Test conversion to JIT after loading."""
        # Setup
        model_path = "/path/to/model.pt"
        mock_model = MagicMock()
        torch_mock.load.return_value = mock_model
        mock_traced_model = MagicMock()
        torch_mock.jit.trace.return_value = mock_traced_model
        
        # Configure to use JIT for regular model
        self.model.config.use_jit = True
        
        # Make isinstance return True for nn.Module
        torch_mock.nn.Module.__instancecheck__ = MagicMock(return_value=True)
        
        # Execute
        with patch("os.path.exists", return_value=True):
            result = self.model.load(model_path)
        
        # Assert
        self.assertTrue(result)
        self.assertTrue(self.model.is_loaded)
        torch_mock.rand.assert_called_once()  # Called to create dummy input
        torch_mock.jit.trace.assert_called_once()  # Traced the model
        self.assertEqual(self.model._model, mock_traced_model)  # Model is now traced
        
    def test_predict_numpy_input(self):
        """Test prediction with numpy array input."""
        # Setup
        mock_model = MagicMock()
        mock_model.return_value = tensor_mock
        self.model._model = mock_model
        self.model.is_loaded = True
        
        # Execute
        inputs = np.array([[1, 2, 3]])
        result = self.model.predict(inputs)
        
        # Assert
        torch_mock.tensor.assert_called_with(inputs, device=self.model._device)
        mock_model.assert_called_once()
        self.assertTrue(isinstance(result, np.ndarray))
        
    def test_predict_tensor_input(self):
        """Test prediction with tensor input."""
        # Setup
        mock_model = MagicMock()
        mock_model.return_value = tensor_

