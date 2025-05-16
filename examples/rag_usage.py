#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG Pipeline Example

This example demonstrates how to set up and use the Retrieval-Augmented Generation (RAG)
pipeline in AutoAI-AgentRAG. It shows how to connect different knowledge sources,
retrieve information, and integrate it with an agent.
"""

import logging
import sys
import os
from typing import Dict, Any, List

from autoai_agentrag import Agent, RAGPipeline
from autoai_agentrag.rag.sources import KnowledgeSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Create a simple file-based knowledge source for demonstration
class FileKnowledgeSource(KnowledgeSource):
    """A simple file-based knowledge source for demonstration purposes."""
    
    def __init__(self, name: str, file_path: str):
        """
        Initialize the file knowledge source.
        
        Args:
            name: Name for the knowledge source
            file_path: Path to the text file containing knowledge
        """
        super().__init__(name)
        self.file_path = file_path
        self.documents = []
        self._load_documents()
        
    def _load_documents(self):
        """Load documents from the file and parse into a list of documents."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split the content into paragraphs and create documents
            paragraphs = content.split('\n\n')
            self.documents = [
                {"id": f"doc_{i}", "content": para.strip(), "source": self.file_path}
                for i, para in enumerate(paragraphs)
                if para.strip()
            ]
            logger.info(f"Loaded {len(self.documents)} documents from {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to load documents from {self.file_path}: {str(e)}")
            self.documents = []
            
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for documents relevant to the query.
        
        This is a simplified implementation that performs basic keyword matching.
        In a real application, you would use vector embeddings and similarity search.
        
        Args:
            query: The search query
            **kwargs: Additional parameters like top_k
            
        Returns:
            A list of relevant documents
        """
        top_k = kwargs.get('top_k', 5)
        query_terms = query.lower().split()
        
        # Simple keyword matching (not efficient for production use)
        scored_docs = []
        for doc in self.documents:
            content = doc["content"].lower()
            score = sum(1 for term in query_terms if term in content)
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: The document ID
            
        Returns:
            The document if found, or None
        """
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None


def create_sample_knowledge_file():
    """Create a sample knowledge file for demonstration purposes."""
    sample_file = "sample_knowledge.txt"
    content = """
Artificial Intelligence (AI) is the field of computer science dedicated to creating systems capable of performing tasks that typically require human intelligence.

Machine Learning is a subset of AI focused on building systems that learn from data, rather than being explicitly programmed.

Deep Learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to analyze various factors of data.

Retrieval-Augmented Generation (RAG) is an AI framework that combines retrieval-based methods with generative models to create more accurate and informed outputs.

The RAG pipeline first retrieves relevant documents from a knowledge base, then uses this context to augment the generation process.

TensorFlow is an open-source machine learning framework developed by Google. It's widely used for training neural networks and other statistical models.

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It's known for its flexibility and dynamic computational graph.

Natural Language Processing (NLP) is a field of AI focused on enabling computers to understand, interpret, and generate human language.

Computer Vision is a field of AI that trains computers to interpret and understand the visual world, processing and analyzing digital images or videos.

Reinforcement Learning is a type of machine learning where an agent learns to make decisions by performing actions and receiving rewards or penalties.
"""
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return os.path.abspath(sample_file)


def main():
    """
    Demonstrate RAG pipeline usage.
    
    This function shows how to:
    1. Set up a RAG pipeline with different knowledge sources
    2. Retrieve information from the sources
    3. Connect the RAG pipeline to an agent
    4. Execute tasks with RAG-augmented context
    """
    try:
        # Create a sample knowledge file
        sample_file_path = create_sample_knowledge_file()
        logger.info(f"Created sample knowledge file at: {sample_file_path}")
        
        # Initialize the RAG pipeline
        logger.info("Creating RAG pipeline")
        rag_pipeline = RAGPipeline(name="demo-pipeline")
        
        # Add a file-based knowledge source
        file_source = FileKnowledgeSource("ai-concepts", sample_file_path)
        rag_pipeline.add_source("ai-concepts", file_source)
        
        # Add a mock web API source (for demonstration)
        # In a real application, you would use a proper web API source
        try:
            from autoai_agentrag.rag.sources import WebAPISource
            web_source = WebAPISource("web-api", "https://example.com/api")
            rag_pipeline.add_source("web-api", web_source)
        except (ImportError, AttributeError):
            logger.warning("WebAPISource not available or not fully implemented yet. Skipping.")
        
        # Perform retrieval using the RAG pipeline
        logger.info("\nPerforming retrieval using the RAG pipeline")
        query = "What is the relationship between machine learning and AI?"
        results = rag_pipeline.retrieve(query, top_k=3)
        
        # Display retrieval results
        logger.info(f"Query: {query}")
        for source_name, documents in results.items():
            logger.info(f"\nSource: {source_name}")
            for i, doc in enumerate(documents):
                logger.info(f"Document {i+1}: {doc.get('content', '')[:100]}...")
        
        # Connect the RAG pipeline to an agent
        logger.info("\nConnecting RAG pipeline to an agent")
        agent = Agent(name="rag-enabled-agent")
        agent.connect_rag(rag_pipeline)
        
        # Execute a query that benefits from RAG context
        logger.info("Executing task with RAG-augmented context")
        result = agent.execute("Explain the difference between deep learning and machine learning")
        
        # Display the result
        if result.success:
            logger.info(f"Task completed successfully")
            logger.info(f"Result: {result.result}")
        else:
            logger.error(f"Task failed: {result.error}")
        
        # Clean up the sample file
        try:
            os.remove(sample_file_path)
            logger.info(f"Removed sample knowledge file: {sample_file_path}")
        except OSError as e:
            logger.warning(f"Failed to remove sample file: {str(e)}")
        
        logger.info("RAG pipeline demonstration completed")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

