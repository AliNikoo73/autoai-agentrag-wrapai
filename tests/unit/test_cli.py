"""
Unit tests for the CLI functionality in AutoAI-AgentRAG.

This module contains tests for the CLI commands and options.
"""

import os
import sys
import unittest
from unittest.mock import patch, mock_open, MagicMock, call

import click
from click.testing import CliRunner

from autoai_agentrag.cli.main import cli, init, train, deploy, setup_logging


class TestCLIBase(unittest.TestCase):
    """Base class for CLI tests."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.runner = CliRunner()


class TestMainCLI(TestCLIBase):
    """Test cases for the main CLI group."""
    
    def test_version_command(self):
        """Test the --version option."""
        from autoai_agentrag import __version__
        
        result = self.runner.invoke(cli, ['--version'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn(__version__, result.output)
    
    @patch('autoai_agentrag.cli.main.setup_logging')
    def test_verbose_flag(self, mock_setup_logging):
        """Test the --verbose flag."""
        result = self.runner.invoke(cli, ['--verbose'])
        
        # Should exit with code 0 even though no command is specified
        # because it's just showing help text
        self.assertEqual(result.exit_code, 0)
        
        # setup_logging should be called with verbose=True
        mock_setup_logging.assert_called_once_with(True)
    
    def test_help_text(self):
        """Test the help text."""
        result = self.runner.invoke(cli, ['--help'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('AutoAI-AgentRAG CLI', result.output)
        self.assertIn('--version', result.output)
        self.assertIn('--verbose', result.output)
        self.assertIn('--help', result.output)
        

class TestInitCommand(TestCLIBase):
    """Test cases for the init command."""
    
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_init_basic_template(self, mock_file, mock_exists, mock_makedirs):
        """Test initializing a project with the basic template."""
        # Mock project directory doesn't exist
        mock_exists.return_value = False
        
        result = self.runner.invoke(cli, ['init', 'my-project'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Creating new AutoAI-AgentRAG project: my-project', result.output)
        self.assertIn('Project initialized successfully', result.output)
        
        # Check directory creation
        mock_makedirs.assert_any_call(os.path.join('my-project'), exist_ok=True)
        mock_makedirs.assert_any_call(os.path.join('my-project', 'models'), exist_ok=True)
        mock_makedirs.assert_any_call(os.path.join('my-project', 'data'), exist_ok=True)
        mock_makedirs.assert_any_call(os.path.join('my-project', 'configs'), exist_ok=True)
        mock_makedirs.assert_any_call(os.path.join('my-project', 'agents'), exist_ok=True)
        
        # Check file creation for basic template
        mock_file.assert_any_call(os.path.join('my-project', 'requirements.txt'), 'w')
        mock_file.assert_any_call(os.path.join('my-project', 'run.py'), 'w')
        mock_file.assert_any_call(os.path.join('my-project', 'README.md'), 'w')
        mock_file.assert_any_call(os.path.join('my-project', 'configs', 'config.yaml'), 'w')
    
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_init_full_template(self, mock_file, mock_exists, mock_makedirs):
        """Test initializing a project with the full template."""
        # Mock project directory doesn't exist
        mock_exists.return_value = False
        
        result = self.runner.invoke(cli, ['init', 'my-project', '--template', 'full'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Creating new AutoAI-AgentRAG project: my-project', result.output)
        self.assertIn('Project initialized successfully', result.output)
        
        # Check additional files for full template
        mock_file.assert_any_call(os.path.join('my-project', 'Dockerfile'), 'w')
        mock_file.assert_any_call(os.path.join('my-project', 'docker-compose.yml'), 'w')
        mock_file.assert_any_call(os.path.join('my-project', 'agents', 'custom_agent.py'), 'w')
        mock_file.assert_any_call(os.path.join('my-project', 'notebooks', 'quickstart.ipynb'), 'w')
    
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_init_minimal_template(self, mock_file, mock_exists, mock_makedirs):
        """Test initializing a project with the minimal template."""
        # Mock project directory doesn't exist
        mock_exists.return_value = False
        
        result = self.runner.invoke(cli, ['init', 'my-project', '--template', 'minimal'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Creating new AutoAI-AgentRAG project: my-project', result.output)
        self.assertIn('Project initialized successfully', result.output)
        
        # Check minimal files
        mock_file.assert_any_call(os.path.join('my-project', 'requirements.txt'), 'w')
        mock_file.assert_any_call(os.path.join('my-project', 'run.py'), 'w')
        mock_file.assert_any_call(os.path.join('my-project', 'README.md'), 'w')
    
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('click.confirm')
    def test_init_existing_directory(self, mock_confirm, mock_file, mock_exists, mock_makedirs):
        """Test initializing a project in an existing directory."""
        # Mock project directory exists
        mock_exists.return_value = True
        # Mock user confirms overwrite
        mock_confirm.return_value = True
        
        result = self.runner.invoke(cli, ['init', 'my-project'])
        
        self.assertEqual(result.exit_code, 0)
        mock_confirm.assert_called_once()
        self.assertIn('Overwrite?', mock_confirm.call_args[0][0])
    
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_init_custom_directory(self, mock_file, mock_exists, mock_makedirs):
        """Test initializing a project in a custom directory."""
        # Mock project directory doesn't exist
        mock_exists.return_value = False
        
        result = self.runner.invoke(cli, ['init', 'my-project', '--directory', '/custom/path'])
        
        self.assertEqual(result.exit_code, 0)
        # Check directory creation in custom path
        mock_makedirs.assert_any_call(os.path.join('/custom/path', 'my-project'), exist_ok=True)


class TestTrainCommand(TestCLIBase):
    """Test cases for the train command."""
    
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_train_basic_usage(self, mock_file, mock_exists, mock_makedirs):
        """Test basic usage of the train command."""
        # Mock data path exists
        mock_exists.return_value = True
        
        result = self.runner.invoke(cli, [
            'train',
            '--model', 'text-classification',
            '--data', './data'
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Training model: text-classification', result.output)
        self.assertIn('Using data from: ./data', result.output)
        
        # Check output directory creation
        mock_makedirs.assert_called_with('./models', exist_ok=True)
    
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_train_with_custom_parameters(self, mock_file, mock_exists, mock_makedirs):
        """Test train command with custom parameters."""
        # Mock data path exists
        mock_exists.return_value = True
        
        result = self.runner.invoke(cli, [
            'train',
            '--model', 'text-classification',
            '--data', './data',
            '--output', './custom_output',
            '--epochs', '20',
            '--batch-size', '64'
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Training model: text-classification', result.output)
        self.assertIn('Using data from: ./data', result.output)
        self.assertIn('Output directory: ./custom_output', result.output)
        self.assertIn('epochs=20', result.output)
        self.assertIn('batch_size=64', result.output)
        
        # Check custom output directory creation
        mock_makedirs.assert_called_with('./custom_output', exist_ok=True)
    
    def test_train_missing_model(self):
        """Test train command with missing model parameter."""
        result = self.runner.invoke(cli, [
            'train',
            '--data', './data'
        ])
        
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Error: Missing option', result.output)
        self.assertIn('--model', result.output)
    
    def test_train_missing_data(self):
        """Test train command with missing data parameter."""
        result = self.runner.invoke(cli, [
            'train',
            '--model', 'text-classification'
        ])
        
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Error: Missing option', result.output)
        self.assertIn('--data', result.output)
    
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_train_creates_dummy_model_file(self, mock_file, mock_exists, mock_makedirs):
        """Test that train command creates a dummy model file (for demo purposes)."""
        # Mock data path exists
        mock_exists.return_value = True
        
        result = self.runner.invoke(cli, [
            'train',
            '--model', 'text-classification',
            '--data', './data'
        ])
        
        self.assertEqual(result.exit_code, 0)
        
        # Check model file creation
        mock_file.assert_any_call(os.path.join('./models', 'text-classification_trained.pkl'), 'wb')


class TestDeployCommand(TestCLIBase):
    """Test cases for the deploy command."""
    
    def test_deploy_basic_usage(self):
        """Test basic usage of the deploy command."""
        result = self.runner.invoke(cli, [
            'deploy',
            '--agent', 'my-agent'
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Deploying agent: my-agent', result.output)
        self.assertIn('Server will run at: http://127.0.0.1:8000', result.output)
    
    def test_deploy_with_custom_parameters(self):
        """Test deploy command with custom parameters."""
        result = self.runner.invoke(cli, [
            'deploy',
            '--agent', 'my-agent',
            '--port', '9000',
            '--host', '0.0.0.0',
            '--rag'
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Deploying agent: my-agent', result.output)
        self.assertIn('Server will run at: http://0.0.0.0:9000', result.output)
        self.assertIn('RAG capabilities enabled', result.output)
    
    @patch('os.path.exists')
    def test_deploy_with_models_dir(self, mock_exists):
        """Test deploy command with custom models directory."""
        # Mock models directory exists
        mock_exists.return_value = True
        
        result = self.runner.invoke(cli, [
            'deploy',
            '--agent', 'my-agent',
            '--models-dir', './custom_models'
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn('Deploying agent: my-agent', result.output)
    
    def test_deploy_missing_agent(self):
        """Test deploy command with missing agent parameter."""
        result = self.runner.invoke(cli, [
            'deploy'
        ])
        
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Error: Missing option', result.output)
        self.assertIn('--agent', result.output)


@patch('logging.basicConfig')
class TestLogging(unittest.TestCase):
    """Test cases for logging configuration."""
    
    def test_setup_logging_default(self, mock_logging_config):
        """Test default logging setup."""
        setup_logging(verbose=False)
        
        mock_logging_config.assert_called_once()
        args, kwargs = mock_logging_config.call_args
        self.assertEqual(kwargs['level'], logging.INFO)
    
    def test_setup_logging_verbose(self, mock_logging_config):
        """Test verbose logging setup."""
        setup_logging(verbose=True)
        
        mock_logging_config.assert_called_once()
        args, kwargs = mock_logging_config.call_args
        self.assertEqual(kwargs['level'], logging.DEBUG)


if __name__ == '__main__':
    unittest.main()

