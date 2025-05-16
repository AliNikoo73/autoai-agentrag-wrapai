"""
Type definitions for the Agent subsystem of AutoAI-AgentRAG.

This module contains the type definitions, enumerations, and data models
used throughout the Agent framework.
"""

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Enumeration of supported agent types."""
    
    CONVERSATIONAL = "conversational"
    TASK_ORIENTED = "task_oriented"
    AUTONOMOUS = "autonomous"
    COLLABORATIVE = "collaborative"
    CUSTOM = "custom"


class AgentConfig(BaseModel):
    """Configuration parameters for an Agent."""
    
    max_retries: int = Field(default=3, description="Maximum number of task retry attempts")
    timeout_seconds: int = Field(default=60, description="Task timeout in seconds")
    allow_external_calls: bool = Field(default=False, description="Whether the agent can make external API calls")
    memory_size: int = Field(default=10, description="Number of previous interactions to remember")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "allow"


class AgentState(BaseModel):
    """Current operational state of an Agent."""
    
    status: str = Field(
        default="idle", 
        description="Current status (idle, running, error)"
    )
    last_active: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp of last activity"
    )
    task_count: int = Field(
        default=0,
        description="Number of tasks processed"
    )
    error_count: int = Field(
        default=0,
        description="Number of errors encountered"
    )
    
    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True


class TaskResult(BaseModel):
    """Result of a task execution by an Agent."""
    
    agent_id: str = Field(..., description="ID of the agent that executed the task")
    task_id: str = Field(default_factory=lambda: str(id({})), description="Unique ID for this task")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="When the task was executed")
    success: bool = Field(..., description="Whether the task was successful")
    result: Optional[Any] = Field(default=None, description="Task execution result data")
    error: Optional[str] = Field(default=None, description="Error message if task failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the task execution")
    
    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True
        
        
class AgentCapability(BaseModel):
    """Defines a specific capability that can be attached to an agent."""
    
    name: str = Field(..., description="Name of the capability")
    description: str = Field(..., description="Description of what the capability does")
    enabled: bool = Field(default=True, description="Whether the capability is currently enabled")
    version: str = Field(default="1.0.0", description="Version of the capability")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for this capability")
    
    class Config:
        """Pydantic configuration."""
        
        extra = "allow"

