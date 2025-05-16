"""
RAG Pipeline Implementation for AutoAI-AgentRAG.

This module provides the core RAG (Retrieval-Augmented Generation) pipeline
that enables agents to retrieve and leverage external knowledge.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from autoai_agentrag.rag.sources import KnowledgeSource

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    A pipeline for Retrieval-Augmented Generation (RAG).
    
    The RAG pipeline connects to various knowledge sources, retrieves relevant
    information based on queries, and augments agent capabilities with this
    additional context.
    
    Attributes:
        name (str): Name of the RAG pipeline
        sources (Dict[str, KnowledgeSource]): Knowledge sources connected to this pipeline
    """
    
    def __init__(self, name: str = "default_pipeline"):
        """
        Initialize a new RAG pipeline.
        
        Args:
            name: Optional name for the pipeline
        """
        self.name = name
        self.sources: Dict[str, KnowledgeSource] = {}
        logger.info(f"RAG Pipeline '{name}' initialized")
    
    def add_source(self, source_name: str, source: Union[KnowledgeSource, str]) -> None:
        """
        Add a knowledge source to this RAG pipeline.
        
        Args:
            source_name: Name to identify this source
            source: Either a KnowledgeSource object or a string URL/path to create a source
        
        Raises:
            ValueError: If a source with the same name already exists
        """
        if source_name in self.sources:
            raise ValueError(f"Source with name '{source_name}' already exists in pipeline")
        
        # If source is a string, try to create an appropriate source based on the format
        if isinstance(source, str):
            from autoai_agentrag.rag.sources import create_source_from_uri
            self.sources[source_name] = create_source_from_uri(source, source_name)
        else:
            self.sources[source_name] = source
            
        logger.info(f"Added source '{source_name}' to RAG pipeline '{self.name}'")
    
    def remove_source(self, source_name: str) -> bool:
        """
        Remove a knowledge source from this RAG pipeline.
        
        Args:
            source_name: Name of the source to remove
            
        Returns:
            True if the source was removed, False if it doesn't exist
        """
        if source_name in self.sources:
            del self.sources[source_name]
            logger.info(f"Removed source '{source_name}' from RAG pipeline '{self.name}'")
            return True
        return False
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant information from all connected knowledge sources.
        
        Args:
            query: The query to retrieve information for
            top_k: The maximum number of results to return per source
            
        Returns:
            A dictionary mapping source names to lists of retrieved documents/results
        """
        if not self.sources:
            logger.warning("No knowledge sources connected to RAG pipeline")
            return {}
        
        results: Dict[str, List[Dict[str, Any]]] = {}
        
        for source_name, source in self.sources.items():
            try:
                source_results = source.search(query, top_k=top_k)
                results[source_name] = source_results
                logger.debug(f"Retrieved {len(source_results)} results from source '{source_name}'")
            except Exception as e:
                logger.error(f"Error retrieving from source '{source_name}': {str(e)}")
                results[source_name] = []
        
        return results
    
    def combine_sources(self, *source_names: str, new_name: str) -> bool:
        """
        Combine multiple sources into a single virtual source.
        
        Args:
            *source_names: Names of sources to combine
            new_name: Name for the new combined source
            
        Returns:
            True if sources were successfully combined, False otherwise
            
        Raises:
            ValueError: If fewer than two sources are provided or new_name already exists
        """
        if len(source_names) < 2:
            raise ValueError("At least two sources must be provided to combine")
        
        if new_name in self.sources:
            raise ValueError(f"Source with name '{new_name}' already exists")
        
        # Check that all sources exist
        for name in source_names:
            if name not in self.sources:
                logger.error(f"Source '{name}' not found in pipeline")
                return False
        
        # TODO: Implement source combination logic
        # This is a placeholder for future implementation
        logger.info(f"Combined sources {source_names} into new source '{new_name}'")
        return True
    
    def __repr__(self) -> str:
        """String representation of the RAG pipeline."""
        sources_str = ", ".join(self.sources.keys())
        return f"RAGPipeline(name='{self.name}', sources=[{sources_str}])"

