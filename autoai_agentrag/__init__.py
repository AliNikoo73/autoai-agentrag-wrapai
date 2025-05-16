"""
AutoAI-AgentRAG: An intelligent automation library integrating AI Agents, RAG, and ML.

This package provides tools for building intelligent automation systems that combine
AI agents, retrieval-augmented generation, and machine learning models.
"""

__version__ = "0.1.0"
__author__ = "AutoAI-AgentRAG Team"
__email__ = "info@autoai-agentrag.com"

# Import main package components for easier access
from autoai_agentrag.agent.base import Agent
from autoai_agentrag.rag.pipeline import RAGPipeline
from autoai_agentrag.ml.model import MLModel

# Version information tuple
VERSION_INFO = tuple(map(int, __version__.split('.')))

__all__ = [
    'Agent',
    'RAGPipeline',
    'MLModel',
    '__version__',
    'VERSION_INFO',
]

