"""
Unit tests for the RAG (Retrieval-Augmented Generation) system in AutoAI-AgentRAG.

This module contains tests for the RAG pipeline and knowledge source implementations.
"""

import unittest
import os
from unittest.mock import MagicMock, patch, mock_open

from autoai_agentrag.rag.pipeline import RAGPipeline
from autoai_agentrag.rag.sources import KnowledgeSource


# Define a mock KnowledgeSource implementation for testing
class MockKnowledgeSource(KnowledgeSource):
    """Mock knowledge source for testing."""
    
    def __init__(self, name, data=None):
        """Initialize with optional mock data."""
        super().__init__(name)
        self.data = data or []
        self.search_called = False
        self.get_document_called = False
        
    def search(self, query, **kwargs):
        """Mock search method."""
        self.search_called = True
        self.last_query = query
        self.last_kwargs = kwargs
        return self.data
    
    def get_document(self, doc_id):
        """Mock document retrieval."""
        self.get_document_called = True
        self.last_doc_id = doc_id
        for doc in self.data:
            if doc.get('id') == doc_id:
                return doc
        return None


class TestKnowledgeSource(unittest.TestCase):
    """Test cases for the KnowledgeSource base class and implementations."""
    
    def test_knowledge_source_init(self):
        """Test initialization of a KnowledgeSource."""
        source = MockKnowledgeSource("test-source")
        self.assertEqual(source.name, "test-source")
        
    def test_search_functionality(self):
        """Test the search functionality of a KnowledgeSource."""
        mock_data = [
            {"id": "doc1", "content": "This is document 1"},
            {"id": "doc2", "content": "This is document 2"}
        ]
        source = MockKnowledgeSource("test-source", mock_data)
        
        results = source.search("test query", top_k=5)
        
        self.assertTrue(source.search_called)
        self.assertEqual(source.last_query, "test query")
        self.assertEqual(source.last_kwargs, {"top_k": 5})
        self.assertEqual(results, mock_data)
        
    def test_get_document(self):
        """Test retrieving a specific document by ID."""
        mock_data = [
            {"id": "doc1", "content": "This is document 1"},
            {"id": "doc2", "content": "This is document 2"}
        ]
        source = MockKnowledgeSource("test-source", mock_data)
        
        # Get existing document
        doc = source.get_document("doc1")
        self.assertTrue(source.get_document_called)
        self.assertEqual(source.last_doc_id, "doc1")
        self.assertEqual(doc, mock_data[0])
        
        # Get non-existent document
        doc = source.get_document("doc3")
        self.assertIsNone(doc)
        

class TestRAGPipeline(unittest.TestCase):
    """Test cases for the RAGPipeline class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.pipeline = RAGPipeline(name="test-pipeline")
        
        # Create some mock sources
        self.source1 = MockKnowledgeSource(
            "source1", 
            [{"id": "doc1", "content": "Content from source 1"}]
        )
        self.source2 = MockKnowledgeSource(
            "source2", 
            [{"id": "doc2", "content": "Content from source 2"}]
        )
        
    def test_pipeline_initialization(self):
        """Test that RAGPipeline initializes with correct values."""
        self.assertEqual(self.pipeline.name, "test-pipeline")
        self.assertEqual(len(self.pipeline.sources), 0)
        
    def test_add_source(self):
        """Test adding a knowledge source to the pipeline."""
        self.pipeline.add_source("source1", self.source1)
        
        self.assertEqual(len(self.pipeline.sources), 1)
        self.assertIn("source1", self.pipeline.sources)
        self.assertEqual(self.pipeline.sources["source1"], self.source1)
        
    def test_add_duplicate_source(self):
        """Test that adding a duplicate source raises ValueError."""
        self.pipeline.add_source("source1", self.source1)
        
        with self.assertRaises(ValueError):
            self.pipeline.add_source("source1", self.source2)
            
    def test_remove_source(self):
        """Test removing a knowledge source from the pipeline."""
        self.pipeline.add_source("source1", self.source1)
        self.pipeline.add_source("source2", self.source2)
        
        result = self.pipeline.remove_source("source1")
        
        self.assertTrue(result)
        self.assertEqual(len(self.pipeline.sources), 1)
        self.assertNotIn("source1", self.pipeline.sources)
        self.assertIn("source2", self.pipeline.sources)
        
    def test_remove_nonexistent_source(self):
        """Test removing a source that doesn't exist returns False."""
        result = self.pipeline.remove_source("nonexistent")
        self.assertFalse(result)
        
    def test_retrieve_with_sources(self):
        """Test retrieving from multiple sources."""
        self.pipeline.add_source("source1", self.source1)
        self.pipeline.add_source("source2", self.source2)
        
        results = self.pipeline.retrieve("test query", top_k=5)
        
        # Check that search was called on both sources
        self.assertTrue(self.source1.search_called)
        self.assertTrue(self.source2.search_called)
        
        # Check that the results contain data from both sources
        self.assertIn("source1", results)
        self.assertIn("source2", results)
        self.assertEqual(results["source1"], self.source1.data)
        self.assertEqual(results["source2"], self.source2.data)
        
    def test_retrieve_without_sources(self):
        """Test retrieving when no sources are connected."""
        results = self.pipeline.retrieve("test query")
        
        self.assertEqual(results, {})
        
    def test_retrieve_with_source_error(self):
        """Test retrieving when a source raises an exception."""
        # Create a source that raises an exception
        error_source = MockKnowledgeSource("error_source")
        error_source.search = MagicMock(side_effect=Exception("Test error"))
        
        self.pipeline.add_source("source1", self.source1)
        self.pipeline.add_source("error_source", error_source)
        
        results = self.pipeline.retrieve("test query")
        
        # Results should still contain data from the working source
        self.assertIn("source1", results)
        self.assertEqual(results["source1"], self.source1.data)
        
        # The error source should return an empty list
        self.assertIn("error_source", results)
        self.assertEqual(results["error_source"], [])
        
    def test_combine_sources(self):
        """Test combining multiple sources."""
        self.pipeline.add_source("source1", self.source1)
        self.pipeline.add_source("source2", self.source2)
        
        # This is a stub test since combine_sources is a placeholder in the implementation
        result = self.pipeline.combine_sources("source1", "source2", new_name="combined")
        
        # The current implementation just logs and returns True
        self.assertTrue(result)
        
    def test_combine_sources_validation(self):
        """Test validation in the combine_sources method."""
        self.pipeline.add_source("source1", self.source1)
        
        # Test with insufficient sources
        with self.assertRaises(ValueError):
            self.pipeline.combine_sources("source1", new_name="combined")
            
        # Test with nonexistent source
        result = self.pipeline.combine_sources("source1", "nonexistent", new_name="combined")
        self.assertFalse(result)
        
        # Test with duplicate name
        self.pipeline.add_source("combined", self.source2)
        with self.assertRaises(ValueError):
            self.pipeline.combine_sources("source1", "combined", new_name="combined")
            
    def test_repr(self):
        """Test the string representation of RAGPipeline."""
        self.pipeline.add_source("source1", self.source1)
        self.pipeline.add_source("source2", self.source2)
        
        expected = "RAGPipeline(name='test-pipeline', sources=[source1, source2])"
        self.assertEqual(repr(self.pipeline), expected)


@patch('autoai_agentrag.rag.sources.create_source_from_uri')
class TestRAGPipelineWithStringSource(unittest.TestCase):
    """Test RAGPipeline with string source creation."""
    
    def test_add_source_from_string(self, mock_create_source):
        """Test adding a source from a string URI."""
        # Setup
        pipeline = RAGPipeline()
        mock_source = MockKnowledgeSource("web_source")
        mock_create_source.return_value = mock_source
        
        # Execute
        pipeline.add_source("web_source", "https://example.com/api")
        
        # Assert
        mock_create_source.assert_called_once_with("https://example.com/api", "web_source")
        self.assertEqual(pipeline.sources["web_source"], mock_source)


if __name__ == '__main__':
    unittest.main()

