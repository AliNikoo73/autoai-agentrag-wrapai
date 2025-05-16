"""
Unit tests for the Agent functionality in AutoAI-AgentRAG.

This module contains tests for the Agent base class and related types.
"""

import uuid
import unittest
from unittest.mock import MagicMock, patch

from autoai_agentrag.agent.base import Agent
from autoai_agentrag.agent.types import AgentType, AgentConfig, AgentState, TaskResult
from autoai_agentrag.rag.pipeline import RAGPipeline
from autoai_agentrag.ml.model import MLModel


class TestAgentTypes(unittest.TestCase):
    """Test cases for Agent type classes."""
    
    def test_agent_type_enum(self):
        """Test that AgentType enum contains expected values."""
        self.assertEqual(AgentType.CONVERSATIONAL, "conversational")
        self.assertEqual(AgentType.TASK_ORIENTED, "task_oriented")
        self.assertEqual(AgentType.AUTONOMOUS, "autonomous")
        self.assertEqual(AgentType.COLLABORATIVE, "collaborative")
        self.assertEqual(AgentType.CUSTOM, "custom")
        
    def test_agent_config_defaults(self):
        """Test that AgentConfig has correct default values."""
        config = AgentConfig()
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.timeout_seconds, 60)
        self.assertFalse(config.allow_external_calls)
        self.assertEqual(config.memory_size, 10)
        self.assertFalse(config.verbose)
        
    def test_agent_config_custom_values(self):
        """Test that AgentConfig accepts custom values."""
        config = AgentConfig(
            max_retries=5,
            timeout_seconds=120,
            allow_external_calls=True,
            memory_size=20,
            verbose=True
        )
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.timeout_seconds, 120)
        self.assertTrue(config.allow_external_calls)
        self.assertEqual(config.memory_size, 20)
        self.assertTrue(config.verbose)
        
    def test_agent_state_defaults(self):
        """Test that AgentState has correct default values."""
        state = AgentState()
        self.assertEqual(state.status, "idle")
        self.assertIsNotNone(state.last_active)
        self.assertEqual(state.task_count, 0)
        self.assertEqual(state.error_count, 0)
        
    def test_task_result_creation(self):
        """Test TaskResult creation and properties."""
        agent_id = str(uuid.uuid4())
        result = TaskResult(
            agent_id=agent_id,
            success=True,
            result={"data": "test"},
            error=None
        )
        self.assertEqual(result.agent_id, agent_id)
        self.assertTrue(result.success)
        self.assertEqual(result.result, {"data": "test"})
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.task_id)
        self.assertIsNotNone(result.timestamp)


class TestAgent(unittest.TestCase):
    """Test cases for the Agent base class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent_name = "test-agent"
        self.agent = Agent(name=self.agent_name)
        
    def test_agent_initialization(self):
        """Test that Agent initializes with correct values."""
        self.assertEqual(self.agent.name, self.agent_name)
        self.assertIsNotNone(self.agent.agent_id)
        self.assertEqual(self.agent.agent_type, AgentType.TASK_ORIENTED)
        self.assertIsInstance(self.agent.config, AgentConfig)
        self.assertIsInstance(self.agent.state, AgentState)
        self.assertEqual(self.agent.state.status, "initialized")
        
    def test_agent_custom_type_and_config(self):
        """Test that Agent accepts custom type and config."""
        config = AgentConfig(max_retries=5, verbose=True)
        agent = Agent(
            name="custom-agent",
            agent_type=AgentType.CONVERSATIONAL,
            config=config
        )
        self.assertEqual(agent.name, "custom-agent")
        self.assertEqual(agent.agent_type, AgentType.CONVERSATIONAL)
        self.assertEqual(agent.config.max_retries, 5)
        self.assertTrue(agent.config.verbose)
        
    def test_connect_rag(self):
        """Test connecting a RAG pipeline to an agent."""
        mock_rag = MagicMock(spec=RAGPipeline)
        self.agent.connect_rag(mock_rag)
        self.assertEqual(self.agent._rag_pipeline, mock_rag)
        
    def test_add_model(self):
        """Test adding a model to an agent."""
        mock_model = MagicMock(spec=MLModel)
        mock_model.name = "test-model"
        self.agent.add_model(mock_model)
        self.assertIn("test-model", self.agent._models)
        self.assertEqual(self.agent._models["test-model"], mock_model)
        
    def test_add_model_with_custom_name(self):
        """Test adding a model with a custom name."""
        mock_model = MagicMock(spec=MLModel)
        mock_model.name = "original-name"
        self.agent.add_model(mock_model, model_name="custom-name")
        self.assertIn("custom-name", self.agent._models)
        self.assertEqual(self.agent._models["custom-name"], mock_model)
        
    @patch.object(RAGPipeline, 'retrieve')
    def test_execute_with_rag(self, mock_retrieve):
        """Test executing a task with RAG."""
        # Setup
        mock_rag = MagicMock(spec=RAGPipeline)
        mock_retrieve.return_value = {"source1": [{"content": "test data"}]}
        mock_rag.retrieve = mock_retrieve
        self.agent.connect_rag(mock_rag)
        
        # Execute
        result = self.agent.execute("test query")
        
        # Assert
        self.assertTrue(result.success)
        self.assertEqual(result.agent_id, self.agent.agent_id)
        self.assertIsNotNone(result.result)
        mock_retrieve.assert_called_once_with("test query")
        
    def test_execute_without_rag(self):
        """Test executing a task without RAG."""
        result = self.agent.execute("test query")
        self.assertTrue(result.success)
        self.assertEqual(result.agent_id, self.agent.agent_id)
        self.assertIsNotNone(result.result)
        
    def test_execute_with_models(self):
        """Test executing a task with models loaded."""
        # Add some mock models
        mock_model1 = MagicMock(spec=MLModel)
        mock_model1.name = "model1"
        mock_model2 = MagicMock(spec=MLModel)
        mock_model2.name = "model2"
        
        self.agent.add_model(mock_model1)
        self.agent.add_model(mock_model2)
        
        # Execute
        result = self.agent.execute("test query")
        
        # Assert
        self.assertTrue(result.success)
        self.assertIsNotNone(result.result)
        self.assertIn("models_used", result.result)
        self.assertEqual(len(result.result["models_used"]), 2)
        self.assertIn("model1", result.result["models_used"])
        self.assertIn("model2", result.result["models_used"])
        
    def test_execute_error_handling(self):
        """Test that errors during execution are properly handled."""
        # Create a mock RAG pipeline that raises an exception
        mock_rag = MagicMock(spec=RAGPipeline)
        mock_rag.retrieve.side_effect = Exception("Test error")
        self.agent.connect_rag(mock_rag)
        
        # Execute
        result = self.agent.execute("test query")
        
        # Assert
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertIsNotNone(result.error)
        self.assertIn("Test error", result.error)
        self.assertEqual(self.agent.state.status, "error")
        
    def test_repr(self):
        """Test the string representation of Agent."""
        expected = f"Agent(name='{self.agent_name}', id='{self.agent.agent_id}', type='{self.agent.agent_type}')"
        self.assertEqual(repr(self.agent), expected)


if __name__ == '__main__':
    unittest.main()

