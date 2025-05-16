#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom Agent Example

This example demonstrates how to create a custom agent by extending the base Agent class
with specialized capabilities. It shows how to implement domain-specific functionality,
override methods, and add custom behaviors.
"""

import logging
import sys
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Union

from autoai_agentrag import Agent, RAGPipeline, MLModel
from autoai_agentrag.agent.types import AgentType, AgentConfig, TaskResult, AgentState
from autoai_agentrag.agent.base import Agent as BaseAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class DataAnalysisAgent(BaseAgent):
    """
    A specialized agent for data analysis tasks.
    
    This agent extends the base Agent with additional capabilities for
    processing and analyzing data, handling specific data formats, and
    providing data-specific insights.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[AgentConfig] = None,
        supported_formats: Optional[List[str]] = None
    ):
        """
        Initialize a new DataAnalysisAgent.
        
        Args:
            name: A human-readable name for the agent
            config: Optional configuration parameters
            supported_formats: List of supported data formats (e.g., 'csv', 'json')
        """
        # Initialize the base Agent with TASK_ORIENTED type
        super().__init__(name=name, agent_type=AgentType.TASK_ORIENTED, config=config)
        
        # Add custom attributes
        self.supported_formats = supported_formats or ["csv", "json", "txt"]
        self.analysis_history = []
        
        logger.info(f"DataAnalysisAgent '{name}' initialized with supported formats: {self.supported_formats}")
    
    def validate_data(self, data: Union[str, Dict, List]) -> bool:
        """
        Validate that the provided data is in a supported format.
        
        Args:
            data: The data to validate, either as a file path or a data structure
            
        Returns:
            True if the data is valid, False otherwise
        """
        if isinstance(data, str):
            # Check if it's a file path with a supported extension
            ext = data.split(".")[-1].lower()
            if ext not in self.supported_formats:
                logger.warning(f"Unsupported file format: {ext}. Supported formats: {self.supported_formats}")
                return False
            return True
        elif isinstance(data, (dict, list)):
            # Data structure is already loaded, assume it's valid
            return True
        else:
            logger.warning(f"Unsupported data type: {type(data)}. Expected string path or data structure.")
            return False
        
    def load_data(self, data_path: str) -> Optional[Union[Dict, List]]:
        """
        Load data from a file path.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            The loaded data or None if loading failed
        """
        if not self.validate_data(data_path):
            return None
        
        ext = data_path.split(".")[-1].lower()
        
        try:
            if ext == "json":
                with open(data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif ext == "csv":
                # In a real implementation, you would use pandas or another CSV library
                logger.info("CSV support would use pandas in a real implementation")
                return {"message": "CSV loading placeholder"}
            elif ext == "txt":
                with open(data_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"No loader implementation for {ext} format")
                return None
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {str(e)}")
            return None
    
    def analyze_data(self, data: Any) -> Dict[str, Any]:
        """
        Perform analysis on the provided data.
        
        Args:
            data: The data to analyze
            
        Returns:
            Analysis results
        """
        # This is a simplified placeholder implementation
        # In a real agent, this would use ML models and advanced analytics
        
        results = {"timestamp": time.time()}
        
        if isinstance(data, dict):
            results["type"] = "dictionary"
            results["keys"] = list(data.keys())
            results["key_count"] = len(data)
        elif isinstance(data, list):
            results["type"] = "list"
            results["item_count"] = len(data)
            if data and all(isinstance(item, dict) for item in data):
                # List of dictionaries, like JSON records
                results["record_type"] = "dictionary"
                if data:
                    results["sample_keys"] = list(data[0].keys())
        elif isinstance(data, str):
            results["type"] = "text"
            results["char_count"] = len(data)
            results["word_count"] = len(data.split())
            results["line_count"] = len(data.splitlines())
        else:
            results["type"] = str(type(data))
        
        # Store in analysis history
        self.analysis_history.append({
            "id": str(uuid.uuid4()),
            "timestamp": results["timestamp"],
            "result_summary": f"Analyzed {results.get('type', 'unknown')} data"
        })
        
        return results
    
    def execute(self, task: Union[str, Dict[str, Any]], **kwargs) -> TaskResult:
        """
        Override the base execute method to handle data analysis tasks.
        
        Args:
            task: The task to execute, either as a string path to data or a structured task
            **kwargs: Additional task-specific parameters
            
        Returns:
            A TaskResult object containing the analysis results
        """
        try:
            self.state.status = "running"
            
            # Check if the task is a simple string path to data
            data = None
            if isinstance(task, str) and (task.endswith(".json") or 
                                          task.endswith(".csv") or 
                                          task.endswith(".txt")):
                # Load data from file path
                data = self.load_data(task)
                if data is None:
                    raise ValueError(f"Failed to load data from {task}")
            elif isinstance(task, dict) and "data_path" in task:
                # Load data from the specified path in the task
                data = self.load_data(task["data_path"])
                if data is None:
                    raise ValueError(f"Failed to load data from {task['data_path']}")
            elif isinstance(task, dict) and "data" in task:
                # Use the provided data directly
                data = task["data"]
            else:
                # Fall back to the parent class implementation for other tasks
                return super().execute(task, **kwargs)
            
            # If we have data, analyze it
            if data is not None:
                # Apply RAG if available to augment the analysis
                context = {}
                if self._rag_pipeline is not None:
                    # Create a query from the task or use a default
                    query = task.get("query", "data analysis techniques") if isinstance(task, dict) else "data analysis techniques"
                    context = self._rag_pipeline.retrieve(query)
                
                # Apply ML models if available
                model_results = {}
                for model_name, model in self._models.items():
                    if hasattr(model, "predict") and callable(model.predict):
                        try:
                            # This is a simplified approach - real implementation would be more sophisticated
                            if isinstance(data, (list, dict)):
                                # Skip model prediction for structured data in this example
                                model_results[model_name] = "Structured data prediction skipped in example"
                            elif isinstance(data, str):
                                # Dummy prediction for string data
                                model_results[model_name] = f"Model {model_name} prediction placeholder"
                        except Exception as e:
                            logger.warning(f"Error using model {model_name}: {str(e)}")
                
                # Perform the analysis
                analysis_results = self.analyze_data(data)
                
                # Add additional context to results
                result = {
                    "analysis": analysis_results,
                    "context_used": bool(context),
                    "models_used": list(model_results.keys()),
                    "model_results": model_results
                }
                
                self.state.status = "idle"
                return TaskResult(
                    agent_id=self.agent_id,
                    success=True,
                    result=result,
                    error=None
                )
            else:
                raise ValueError("No data provided for analysis")
                
        except Exception as e:
            logger.error(f"Error executing task with DataAnalysisAgent: {str(e)}")
            self.state.status = "error"
            return TaskResult(
                agent_id=self.agent_id,
                success=False,
                result=None,
                error=str(e)
            )
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of analyses performed by this agent.
        
        Returns:
            List of analysis records
        """
        return self.analysis_history


class NLPAgent(BaseAgent):
    """
    A specialized agent for natural language processing tasks.
    
    This agent focuses on text processing, sentiment analysis,
    named entity recognition, and other NLP-related tasks.
    """
    
    def __init__(
        self,
        name: str,
        languages: Optional[List[str]] = None,
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize a new NLPAgent.
        
        Args:
            name: A human-readable name for the agent
            languages: List of supported languages (ISO codes)
            config: Optional configuration parameters
        """
        # Initialize the base Agent with CONVERSATIONAL type
        super().__init__(name=name, agent_type=AgentType.CONVERSATIONAL, config=config)
        
        # Add custom attributes
        self.languages = languages or ["en", "es", "fr", "de"]
        self.nlp_capabilities = {
            "sentiment_analysis": True,
            "entity_recognition": True,
            "summarization": True,
            "translation": True
        }
        
        logger.info(f"NLPAgent '{name}' initialized with languages: {self.languages}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the provided text.
        
        Args:
            text: The text to analyze
            
        Returns:
            The detected language code
        """
        # This is a placeholder implementation
        # In a real agent, you would use a language detection library
        
        # Simple heuristics for demo purposes
        if "the" in text.lower() or "is" in text.lower():
            return "en"
        elif "el" in text.lower() or "la" in text.lower():
            return "es"
        elif "le" in text.lower() or "la" in text.lower():
            return "fr"
        elif "der" in text.lower() or "die" in text.lower():
            return "de"
        else:
            return "en"  # Default to English
    
    def sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform sentiment analysis on the text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Sentiment analysis results
        """
        # This is a placeholder implementation
        # In a real agent, you would use an NLP model
        
        positive_words = ["good", "great", "excellent", "amazing", "happy", "like", "love"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "sad"]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            sentiment = "positive"
            score = 0.5 + 0.5 * (positive_count / (positive_count + negative_count + 1))
        elif negative_count > positive_count:
            sentiment = "negative"
            score = -0.5 - 0.5 * (negative_count / (positive_count + negative_count + 1))
        else:
            sentiment = "neutral"
            score = 0.0
        
        return {
            "sentiment": sentiment,
            "score": score,
            "positive_words": positive_count,
            "negative_words": negative_count
        }
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        """
        Generate a summary of the text.
        
        Args:
            text: The text to summarize
            max_length: Maximum length of the summary
            
        Returns:
            The summary text
        """
        # This is a placeholder implementation
        # In a real agent, you would use a summarization model
        
        sentences = text.split(".")
        if len(sentences) <= 2:
            return text
        
        # Just return the first 2 sentences as a simple summary
        summary = ". ".join(sentence.strip() for sentence in sentences[:2]) + "."
        
        # Trim to max_length if needed
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."
            
        return summary
    
    def execute(self, task: Union[str, Dict[str, Any]], **kwargs) -> TaskResult:
        """
        Override the base execute method to handle NLP tasks.
        
        Args:
            task: The task to execute, either as a string text or a structured task
            **kwargs: Additional task-specific parameters
            
        Returns:
            A TaskResult object containing the NLP results
        """
        try:
            self.state.status = "running"
            
            # Extract the text and task type
            text = None
            task_type = None
            
            if isinstance(task, str):
                text = task
                task_type = kwargs.get("task_type", "sentiment_analysis")
            elif isinstance(task, dict):
                text = task.get("text", "")
                task_type = task.get("task_type", "sentiment_analysis")
            
            if not text:
                raise ValueError("No text provided for NLP processing")
            
            # Detect the language
            lang = self.detect_language(text)
            
            # Process according to task type
            result = {
                "language": lang,
                "task_type": task_type
            }
            
            if task_type == "sentiment_analysis":
                result["sentiment"] = self.sentiment_analysis(text)
            elif task_type == "summarization":
                max_length = kwargs.get("max_length", 100)
                if isinstance(task, dict) and "max_length" in task:
                    max_length = task["max_length"]
                result["summary"] = self.summarize(text, max_length)
            else:
                # Fall back to the parent class implementation for other task types
                return super().execute(task, **kwargs)
                
            self.state.status = "idle"
            return result
                
        except Exception as e:
            logger.error(f"Error executing task with NLPAgent: {str(e)}")
            self.state.status = "error"
            return TaskResult(
                agent_id=self.agent_id,
                success=False,
                result=None,
                error=str(e)
            )


def main():
    """
    Demonstrate custom agent implementations.
    
    This function shows how to:
    1. Create specialized agent types
    2. Use their custom capabilities
    3. Integrate them with RAG and ML models
    """
    try:
        # Create a DataAnalysisAgent
        logger.info("Creating a DataAnalysisAgent")
        data_agent = DataAnalysisAgent(
            name="data-analysis-agent",
            supported_formats=["json", "csv", "txt"]
        )
        
        # Create sample data for analysis
        sample_data = {
            "users": [
                {"id": 1, "name": "Alice", "age": 30, "active": True},
                {"id": 2, "name": "Bob", "age": 25, "active": False},
                {"id": 3, "name": "Charlie", "age": 35, "active": True}
            ],
            "metadata": {
                "version": "1.0",
                "created_at": "2025-05-15T15:00:00Z"
            }
        }
        
        # Execute a data analysis task with direct data input
        logger.info("Executing data analysis task")
        result = data_agent.execute({"data": sample_data})
        
        # Display the results
        if result.success:
            logger.info("Data analysis completed successfully")
            analysis = result.result["analysis"]
            logger.info(f"Data type: {analysis.get('type')}")
            if "keys" in analysis:
                logger.info(f"Top-level keys: {analysis.get('keys')}")
        else:
            logger.error(f"Data analysis failed: {result.error}")
        
        # Create an NLPAgent
        logger.info("\nCreating an NLPAgent")
        nlp_agent = NLPAgent(
            name="nlp-agent",
            languages=["en", "es", "fr"]
        )
        
        # Execute a sentiment analysis task
        logger.info("Executing sentiment analysis task")
        sample_text = "I really love this product! It's amazing and exceeded my expectations."
        result = nlp_agent.execute({
            "text": sample_text,
            "task_type": "sentiment_analysis"
        })
        
        # Display the results
        if result.success:
            logger.info("Sentiment analysis completed successfully")
            sentiment = result.result.get("sentiment", {})
            logger.info(f"Detected sentiment: {sentiment.get('sentiment')}")
            logger.info(f"Sentiment score: {sentiment.get('score')}")
        else:
            logger.error(f"Sentiment analysis failed: {result.error}")
        
        # Execute a summarization task
        logger.info("\nExecuting summarization task")
        sample_long_text = """
        Artificial Intelligence (AI) has transformed numerous industries over the past decade.
        From healthcare to finance, transportation to entertainment, AI technologies are being
        integrated into systems and processes to enhance efficiency, accuracy, and innovation.
        Machine learning, a subset of AI, enables systems to learn from data and improve over time
        without explicit programming. Deep learning, a further specialized area, uses neural networks
        with multiple layers to analyze various factors of data. As these technologies continue to
        evolve, we can expect even more profound impacts on society and the economy.
        """
        result = nlp_agent.execute({
            "text": sample_long_text,
            "task_type": "summarization",
            "max_length": 150
        })
        
        # Display the results
        if result.success:
            logger.info("Summarization completed successfully")
            summary = result.result.get("summary", "")
            logger.info(f"Summary: {summary}")
        else:
            logger.error(f"Summarization failed: {result.error}")
        
        logger.info("\nCustom agent demonstration completed")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

