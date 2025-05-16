# Examples

This page provides examples of using AutoAI-AgentRAG in various scenarios.

## Basic Usage Examples

### Creating and Using a Basic Agent

This example shows how to create a simple agent and execute a task:

```python
from autoai_agentrag import Agent

# Create a basic agent
agent = Agent("my-agent")

# Execute a simple task
result = agent.execute("Tell me about artificial intelligence")

# Display the result
print(f"Success: {result.success}")
print(f"Result: {result.result}")
```

### Setting Up a RAG Pipeline

This example demonstrates how to create a RAG pipeline and connect it to an agent:

```python
from autoai_agentrag import Agent, RAGPipeline

# Create an agent
agent = Agent("rag-enabled-agent")

# Create a RAG pipeline
pipeline = RAGPipeline()

# Add some knowledge sources
pipeline.add_source("web", "https://example.com/api")
pipeline.add_source("docs", "https://docs.example.com")

# Connect the pipeline to the agent
agent.connect_rag(pipeline)

# Execute a task with RAG-enhanced capabilities
result = agent.execute("What is the capital of France?")
print(f"Result: {result.result}")
```

### Using ML Models

This example shows how to load and use a machine learning model:

```python
from autoai_agentrag import Agent
from autoai_agentrag.ml.model import MLModel

# Load a pre-trained model
model = MLModel.from_pretrained("example-model-path")

# Create an agent
agent = Agent("ml-agent")

# Add the model to the agent
agent.add_model(model)

# Execute a task that uses the model
result = agent.execute("Classify this text")
print(f"Result: {result.result}")
```

## Advanced Use Cases

### Creating a Custom Agent

This example demonstrates how to create a custom agent by extending the base Agent class:

```python
from autoai_agentrag.agent.base import Agent as BaseAgent
from autoai_agentrag.agent.types import AgentType, AgentConfig, TaskResult

class CustomAgent(BaseAgent):
    """A custom agent with specialized capabilities."""
    
    def __init__(self, name, **kwargs):
        """Initialize the custom agent."""
        # Initialize with the CUSTOM agent type
        super().__init__(name, agent_type=AgentType.CUSTOM, **kwargs)
        
        # Add custom attributes
        self.custom_capability = True
    
    def custom_method(self, data):
        """A custom method specific to this agent type."""
        return f"Processed {data} with custom method"
    
    def execute(self, task, **kwargs):
        """Override the execute method to add custom behavior."""
        # Add pre-processing logic
        if isinstance(task, str) and task.startswith("CUSTOM:"):
            # Handle custom task format
            task = task[7:]  # Remove the "CUSTOM:" prefix
        
        # Call the parent execute method for standard processing
        result = super().execute(task, **kwargs)
        
        # Add post-processing logic
        if result.success and result.result:
            result.result["custom_field"] = "Added by custom agent"
        
        return result

# Create and use the custom agent
agent = CustomAgent("my-custom-agent")
result = agent.execute("CUSTOM:process this")
```

### Implementing a Custom Knowledge Source

This example shows how to create a custom knowledge source for the RAG pipeline:

```python
from autoai_agentrag.rag.sources import KnowledgeSource
from typing import List, Dict, Any, Optional

class CustomKnowledgeSource(KnowledgeSource):
    """A custom knowledge source implementation."""
    
    def __init__(self, name, data_path):
        """Initialize with a path to data."""
        super().__init__(name)
        self.data_path = data_path
        self.data = self._load_data()
    
    def _load_data(self):
        """Load data from the specified path."""
        # This is a simplified example - in a real implementation,
        # you would load data from a file, database, or API
        return [
            {"id": "doc1", "content": "This is document 1"},
            {"id": "doc2", "content": "This is document 2"},
            {"id": "doc3", "content": "This is document 3"}
        ]
    
    def search(self, query, **kwargs):
        """Search for documents matching the query."""
        top_k = kwargs.get('top_k', 3)
        
        # Simple keyword matching (not efficient for production)
        results = []
        for doc in self.data:
            if query.lower() in doc["content"].lower():
                results.append(doc)
        
        return results[:top_k]
    
    def get_document(self, doc_id):
        """Retrieve a specific document by ID."""
        for doc in self.data:
            if doc["id"] == doc_id:
                return doc
        return None

# Use the custom knowledge source
from autoai_agentrag import RAGPipeline

pipeline = RAGPipeline()
custom_source = CustomKnowledgeSource("custom", "path/to/data")
pipeline.add_source("custom", custom_source)

# Retrieve information
results = pipeline.retrieve("document")
print(results)
```

## Best Practices

### Error Handling

Always implement proper error handling to make your agents robust:

```python
from autoai_agentrag import Agent

def process_with_agent(query):
    try:
        agent = Agent("error-handling-agent")
        result = agent.execute(query)
        
        if result.success:
            return result.result
        else:
            # Handle task execution failure
            print(f"Task failed: {result.error}")
            return {"error": result.error}
            
    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {str(e)}")
        return {"error": "Internal error"}
```

### Agent Configuration

Configure your agents appropriately for your use case:

```python
from autoai_agentrag import Agent
from autoai_agentrag.agent.types import AgentConfig, AgentType

# For a high-reliability task agent
task_config = AgentConfig(
    max_retries=5,
    timeout_seconds=120,
    allow_external_calls=False,
    verbose=True
)
task_agent = Agent("reliable-task-agent", agent_type=AgentType.TASK_ORIENTED, config=task_config)

# For a conversational agent with external calls
chat_config = AgentConfig(
    max_retries=2,
    timeout_seconds=30,
    allow_external_calls=True,
    memory_size=20
)
chat_agent = Agent("chat-agent", agent_type=AgentType.CONVERSATIONAL, config=chat_config)
```

### Resource Management

Properly manage resources, especially when dealing with ML models:

```python
from autoai_agentrag.ml.tensorflow import TensorFlowModel
import contextlib

@contextlib.contextmanager
def managed_model():
    """Context manager for proper model cleanup."""
    model = TensorFlowModel("temp-model")
    try:
        model.load("/path/to/model")
        yield model
    finally:
        # Perform any cleanup needed
        print("Cleaning up model resources")
        # model.cleanup()  # If there's a cleanup method

# Use the model within the context
with managed_model() as model:
    result = model.predict(some_data)
    # Do something with result
# Model is automatically cleaned up when exiting the context
```

### Logging Configuration

Set up proper logging to help with debugging and monitoring:

```python
import logging
from autoai_agentrag import Agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)

# Create an agent with verbose logging
agent = Agent("logged-agent", config=AgentConfig(verbose=True))

# Execute tasks
agent.execute("Test task")
```

