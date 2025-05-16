"""
Unit tests for the ML model functionality in AutoAI-AgentRAG.

This module contains tests for the ML model base class and related utilities.
"""

import os
import json
import unittest
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import numpy as np

from autoai_agentrag.ml.model import (
    MLModel, ModelFramework, ModelConfig, ModelMetadata, CustomModel
)


# Create a concrete implementation of MLModel for testing
class TestModel(MLModel):
    """Concrete MLModel implementation for testing."""
    
    def __init__(self, name, framework=ModelFramework.CUSTOM, config=None, metadata=None):
        """Initialize the test model."""
        super().__init__(name, framework, config, metadata)
        self.load_called = False
        self.predict_called = False
        self.save_called = False
        self.mock_result = None
        
    def load(self, model_path=None):
        """Mock load implementation."""
        self.load_called = True
        self.load_path = model_path
        self.is_loaded = True
        return True
        
    def predict(self, inputs, **kwargs):
        """Mock predict implementation."""
        self.predict_called = True
        self.predict_inputs = inputs
        self.predict_kwargs = kwargs
        return self.mock_result or np.array([1, 2, 3])
        
    def save(self, save_path):
        """Mock save implementation."""
        self.save_called = True
        self.save_path = save_path
        return True


class TestMLModelBase(unittest.TestCase):
    """Test cases for the MLModel base class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model_name = "test-model"
        self.model = TestModel(name=self.model_name)
        
    def test_model_initialization(self):
        """Test that MLModel initializes with correct values."""
        self.assertEqual(self.model.name, self.model_name)
        self.assertEqual(self.model.framework, ModelFramework.CUSTOM)
        self.assertIsInstance(self.model.config, ModelConfig)
        self.assertIsInstance(self.model.metadata, ModelMetadata)
        self.assertEqual(self.model.metadata.name, self.model_name)
        self.assertEqual(self.model.metadata.framework, ModelFramework.CUSTOM)
        self.assertFalse(self.model.is_loaded)
        
    def test_model_custom_config(self):
        """Test model with custom configuration."""
        config = ModelConfig(batch_size=64, device="cpu")
        model = TestModel(name="custom-config", config=config)
        
        self.assertEqual(model.config.batch_size, 64)
        self.assertEqual(model.config.device, "cpu")
        
    def test_model_custom_metadata(self):
        """Test model with custom metadata."""
        metadata = {
            "name": "custom-metadata",
            "framework": ModelFramework.CUSTOM,
            "version": "1.0.0",
            "author": "Test Author",
            "description": "Test Description",
            "tags": ["test", "ml"]
        }
        model = TestModel(name="custom-metadata", metadata=metadata)
        
        self.assertEqual(model.metadata.name, "custom-metadata")
        self.assertEqual(model.metadata.version, "1.0.0")
        self.assertEqual(model.metadata.author, "Test Author")
        self.assertEqual(model.metadata.description, "Test Description")
        self.assertEqual(model.metadata.tags, ["test", "ml"])
        
    def test_load_method(self):
        """Test the load method."""
        result = self.model.load("test/path/model.h5")
        
        self.assertTrue(result)
        self.assertTrue(self.model.load_called)
        self.assertEqual(self.model.load_path, "test/path/model.h5")
        self.assertTrue(self.model.is_loaded)
        
    def test_predict_method(self):
        """Test the predict method."""
        # Load model first
        self.model.load()
        
        # Run prediction
        inputs = np.array([[1, 2, 3], [4, 5, 6]])
        kwargs = {"batch_size": 2}
        result = self.model.predict(inputs, **kwargs)
        
        self.assertTrue(self.model.predict_called)
        self.assertTrue(np.array_equal(self.model.predict_inputs, inputs))
        self.assertEqual(self.model.predict_kwargs, kwargs)
        self.assertTrue(isinstance(result, np.ndarray))
        
    def test_predict_without_loading(self):
        """Test that predicting without loading raises an error."""
        # Concrete TestModel doesn't enforce is_loaded check, so test with CustomModel
        with patch.object(CustomModel, 'load'):
            model = CustomModel(name="test")
            with self.assertRaises(ValueError):
                model.predict(np.array([1, 2, 3]))
        
    def test_save_method(self):
        """Test the save method."""
        # Load model first
        self.model.load()
        
        result = self.model.save("/tmp/model.h5")
        
        self.assertTrue(result)
        self.assertTrue(self.model.save_called)
        self.assertEqual(self.model.save_path, "/tmp/model.h5")
        
    def test_call_method(self):
        """Test the __call__ method which should invoke predict."""
        # Load model first
        self.model.load()
        
        inputs = np.array([[1, 2, 3]])
        result = self.model(inputs)
        
        self.assertTrue(self.model.predict_called)
        self.assertTrue(np.array_equal(self.model.predict_inputs, inputs))
        
    def test_repr(self):
        """Test the string representation of MLModel."""
        expected = f"MLModel(name='{self.model_name}', framework={self.model.metadata.framework}, status=not loaded)"
        self.assertEqual(repr(self.model), expected)
        
        # Test after loading
        self.model.load()
        expected = f"MLModel(name='{self.model_name}', framework={self.model.metadata.framework}, status=loaded)"
        self.assertEqual(repr(self.model), expected)


class TestModelMetadata(unittest.TestCase):
    """Test cases for model metadata handling."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model = TestModel(name="metadata-test")
        
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_metadata(self, mock_json_dump, mock_file_open):
        """Test saving model metadata to a file."""
        result = self.model.save_metadata("/tmp/model.meta.json")
        
        self.assertTrue(result)
        mock_file_open.assert_called_once_with("/tmp/model.meta.json", "w")
        self.assertTrue(mock_json_dump.called)
        
    @patch("builtins.open", new_callable=mock_open, read_data='{"name": "test", "framework": "custom", "version": "2.0.0"}')
    @patch("json.load", return_value={"name": "test", "framework": "custom", "version": "2.0.0"})
    def test_load_metadata(self, mock_json_load, mock_file_open):
        """Test loading model metadata from a file."""
        metadata = self.model.load_metadata("/tmp/model.meta.json")
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.name, "test")
        self.assertEqual(metadata.framework, ModelFramework.CUSTOM)
        self.assertEqual(metadata.version, "2.0.0")
        mock_file_open.assert_called_once_with("/tmp/model.meta.json", "r")
        self.assertTrue(mock_json_load.called)
        
    @patch("builtins.open", side_effect=IOError("File not found"))
    def test_load_metadata_error(self, mock_file_open):
        """Test error handling when loading metadata."""
        metadata = self.model.load_metadata("/nonexistent/path.json")
        
        self.assertIsNone(metadata)
        mock_file_open.assert_called_once_with("/nonexistent/path.json", "r")


class TestFrameworkDetection(unittest.TestCase):
    """Test cases for framework detection utilities."""
    
    def setUp(self):
        """Create temporary files for testing."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create temporary files with different extensions
        self.tf_path = os.path.join(self.temp_dir, "model.h5")
        self.keras_path = os.path.join(self.temp_dir, "model.keras")
        self.pytorch_path = os.path.join(self.temp_dir, "model.pt")
        self.onnx_path = os.path.join(self.temp_dir, "model.onnx")
        self.sklearn_path = os.path.join(self.temp_dir, "model.joblib")
        
        # Create the files
        for path in [self.tf_path, self.keras_path, self.pytorch_path, self.onnx_path, self.sklearn_path]:
            with open(path, 'w') as f:
                f.write("dummy content")
    
    def tearDown(self):
        """Clean up temporary files."""
        # Remove all temporary files and directory
        for path in [self.tf_path, self.keras_path, self.pytorch_path, self.onnx_path, self.sklearn_path]:
            os.remove(path)
        os.rmdir(self.temp_dir)
    
    def test_infer_framework_from_extension(self):
        """Test inferring framework from file extensions."""
        self.assertEqual(MLModel._infer_framework(self.tf_path), ModelFramework.TENSORFLOW)
        self.assertEqual(MLModel._infer_framework(self.keras_path), ModelFramework.TENSORFLOW)
        self.assertEqual(MLModel._infer_framework(self.pytorch_path), ModelFramework.PYTORCH)
        self.assertEqual(MLModel._infer_framework(self.onnx_path), ModelFramework.ONNX)
        self.assertEqual(MLModel._infer_framework(self.sklearn_path), ModelFramework.SKLEARN)
        
    def test_infer_framework_from_model_name(self):
        """Test inferring framework from model name patterns."""
        self.assertEqual(MLModel._infer_framework("huggingface/bert-base"), ModelFramework.HUGGINGFACE)
        self.assertEqual(MLModel._infer_framework("google/t5-base"), ModelFramework.HUGGINGFACE)
        self.assertEqual(MLModel._infer_framework("hf://gpt2"), ModelFramework.HUGGINGFACE)
        self.assertEqual(MLModel._infer_framework("custom_model"), ModelFramework.CUSTOM)


class TestCustomModel(unittest.TestCase):
    """Test cases for the CustomModel implementation."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model = CustomModel(name="custom-test")
        
    @patch("pickle.load", return_value=MagicMock())
    @patch("builtins.open", new_callable=mock_open, read_data=b"binary data")
    def test_load_pickled_model(self, mock_file_open, mock_pickle_load):
        """Test loading a pickled model."""
        # Create a temporary path
        model_path = "/tmp/model.pkl"
        with patch("os.path.exists", return_value=True):
            result = self.model.load(model_path)
            
            self.assertTrue(result)
            self.assertTrue(self.model.is_loaded)
            mock_file_open.assert_called_once_with(model_path, "rb")
            self.assertTrue(mock_pickle_load.called)
            
    @patch("pickle.load", side_effect=Exception("Pickle error"))
    @patch("builtins.open", new_callable=mock_open)
    def test_load_error_handling(self, mock_file_open, mock_pickle_load):
        """Test error handling when loading a model."""
        with patch("os.path.exists", return_value=True):
            result = self.model.load("/tmp/model.pkl")
            
            self.assertFalse(result)
            self.assertFalse(self.model.is_loaded)
            
    def test_predict_methods(self):
        """Test different predict methods of CustomModel."""
        # Set up a model with different types of _model
        
        # 1. Model with predict method
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 2, 3])
        with patch.object(CustomModel, '_model', mock_model):
            self.model.is_loaded = True
            result = self.model.predict(np.array([4, 5, 6]))
            mock_model.predict.assert_called_once_with(np.array([4, 5, 6]))
            self.assertTrue(np.array_equal(result, np.array([1, 2, 3])))
            
        # 2. Callable model
        callable_model = MagicMock()
        callable_model.return_value = np.array([4, 5, 6])
        with patch.object(CustomModel, '_model', callable_model):
            self.model.is_loaded = True
            result = self.model.predict(np.array([7, 8, 9]))
            callable_model.assert_called_once_with(np.array([7, 8, 9]))
            self.assertTrue(np.array_equal(result, np.array([4, 5, 6])))
            
        # 3. Custom predict function
        def custom_predict(inputs):
            return inputs * 2
            
        model = CustomModel(name="with-custom-predict", predict_fn=custom_predict)
        model.is_loaded = True
        result = model.predict(np.array([1, 2, 3]))
        self.assertTrue(np.array_equal(result, np.array([2, 4, 6])))
        
        # 4. Invalid model
        invalid_model = "not a callable or model"
        with patch.object(CustomModel, '_model', invalid_model):
            self.model.is_loaded = True
            with self.assertRaises(ValueError):
                self.model.predict(np.array([1, 2, 3]))
                
    @patch("pickle.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_method(self, mock_file_open, mock_pickle_dump):
        """Test saving a model."""
        # Set up a model
        self.model._model = MagicMock()
        self.model.is_loaded = True
        
        with patch("os.makedirs", return_value=None):
            result = self.model.save("/tmp/model.pkl")
            
            self.assertTrue(result)
            mock_file_open.assert_called_once_with("/tmp/model.pkl", "wb")
            self.assertTrue(mock_pickle_dump.called)
            

