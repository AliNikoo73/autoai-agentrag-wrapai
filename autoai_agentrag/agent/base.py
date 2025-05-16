"""
Base Agent Implementation for AutoAI-AgentRAG.

This module defines the core Agent class that serves as the foundation for 
creating intelligent automation agents with RAG capabilities.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from autoai_agentrag.agent.types import AgentConfig, AgentState, AgentType, TaskResult
from autoai_agentrag.ml.model import MLModel
from autoai_agentrag.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class Agent:
    """
    The core Agent class for building intelligent automation agents.
    
    Agents can be configured with different capabilities, connect to RAG systems,
    and utilize ML models to perform complex tasks.
    
    Attributes:
        name (str): The name of the agent
        agent_id (str): Unique identifier for the agent
        agent_type (AgentType): The type of agent (e.g., CONVERSATIONAL, TASK_ORIENTED)
        config (AgentConfig): Configuration parameters for the agent
        state (AgentState): Current state of the agent
    """
    
    def __init__(
        self, 
        name: str, 
        agent_type: AgentType = AgentType.TASK_ORIENTED,
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize a new Agent instance.
        
        Args:
            name: A human-readable name for the agent
            agent_type: The type of agent functionality to enable
            config: Optional configuration parameters
        """
        self.name = name
        self.agent_id = str(uuid.uuid4())
        self.agent_type = agent_type
        self.config = config or AgentConfig()
        self.state = AgentState(status="initialized")
        
        self._rag_pipeline: Optional[RAGPipeline] = None
        self._models: Dict[str, MLModel] = {}
        
        logger.info(f"Agent '{name}' (ID: {self.agent_id}) initialized with type {agent_type}")
    
    def connect_rag(self, rag_pipeline: RAGPipeline) -> None:
        """
        Connect a RAG pipeline to this agent for knowledge retrieval.
        
        Args:
            rag_pipeline: The RAG pipeline to connect to this agent
        """
        self._rag_pipeline = rag_pipeline
        logger.info(f"Agent '{self.name}' connected to RAG pipeline")
    
    def add_model(self, model: MLModel, model_name: Optional[str] = None) -> None:
        """
        Add a machine learning model to this agent.
        
        Args:
            model: The machine learning model to add
            model_name: Optional custom name for the model. If not provided, 
                       the model's default name will be used.
        """
        name = model_name or model.name
        self._models[name] = model
        logger.info(f"Model '{name}' added to agent '{self.name}'")
    
    def execute(self, task: Union[str, Dict[str, Any]], **kwargs) -> TaskResult:
        """
        Execute a task using this agent.
        
        Args:
            task: The task to execute, either as a string command or a structured task dict
            **kwargs: Additional task-specific parameters
            
        Returns:
            A TaskResult object containing the execution results and metadata
        """
        try:
            self.state.status = "running"
            
            # If RAG is connected, use it to augment the task with knowledge
            context = {}
            if self._rag_pipeline is not None:
                if isinstance(task, str):
                    context = self._rag_pipeline.retrieve(task)
                else:
                    context = self._rag_pipeline.retrieve(str(task))
            
            # TODO: Implement task execution logic using connected models
            # This is a placeholder implementation
            result = {
                "task": task,
                "status": "completed",
                "context_used": bool(context),
                "models_used": list(self._models.keys())
            }
            
            self.state.status = "idle"
            return TaskResult(
                agent_id=self.agent_id,
                success=True,
                result=result,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Error executing task with agent '{self.name}': {str(e)}")
            self.state.status = "error"
            return TaskResult(
                agent_id=self.agent_id,
                success=False,
                result=None,
                error=str(e)
            )
    
    def __repr__(self) -> str:
        """String representation of the Agent."""
        return f"Agent(name='{self.name}', id='{self.agent_id}', type='{self.agent_type}')"

