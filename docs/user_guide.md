# User Guide

This guide provides detailed information about using AutoAI-AgentRAG for building intelligent automation systems.

## Table of Contents

- [Understanding the Core Components](#understanding-the-core-components)
  - [Agent Framework](#agent-framework)
  - [RAG System](#rag-system)
  - [ML Models](#ml-models)
- [Working with Agents](#working-with-agents)
  - [Creating Agents](#creating-agents)
  - [Agent Configuration](#agent-configuration)
  - [Task Execution](#task-execution)
- [Working with RAG](#working-with-rag)
  - [Setting Up a RAG Pipeline](#setting-up-a-rag-pipeline)
  - [Adding Knowledge Sources](#adding-knowledge-sources)
  - [Retrieving Information](#retrieving-information)
- [Working with ML Models](#working-with-ml-models)
  - [Loading Models](#loading-models)
  - [Making Predictions](#making-predictions)
  - [Framework-Specific Models](#framework-specific-models)
- [Command-Line Interface](#command-line-interface)
  - [Project Management](#project-management)
  - [Model Training](#model-training)
  - [Agent Deployment](#agent-deployment)
- [Advanced Topics](#advanced-topics)
  - [Custom Agents](#custom-agents)
  - [Custom Knowledge Sources](#custom-knowledge-sources)
  - [Plugin Development](#plugin-development)

## Understanding the Core Components

### Agent Framework

The Agent framework provides the foundation for creating AI agents that can perform tasks autonomously. Agents can connect to knowledge sources via RAG and use ML models for predictions.

```python
from autoai_agentrag import Agent
from autoai_agentrag.agent.types import AgentType, AgentConfig

# Create a conversational agent with custom configuration
config = AgentConfig(
    max_retries=5,
    timeout_seconds=60,
    allow_external_calls=True
)
agent = Agent("my-agent", agent_type=AgentType.CONVERSATIONAL, config=config)
```

### RAG System

The RAG (Retrieval-Augmented Generation) system allows agents to access external knowledge sources and use the retrieved information to enhance their capabilities.

```python
from autoai_agentrag import RAGPipeline
from autoai_agentrag.rag.sources import WebAPISource

# Create a RAG pipeline
pipeline = RAGPipeline()

# Add a web API as a knowledge source
web_source = WebAPISource("example-api", "https://api.example.com/v1")
pipeline.add_source("web", web_source)

# You can also add a source directly using a URL
pipeline.add_source("docs", "https://docs.example.com/api")
```

### ML Models

The ML model system provides a unified interface for working with machine learning models from different frameworks (TensorFlow, PyTorch, etc.).

```python
from autoai_agentrag.ml.model import MLModel
from autoai_agentrag.ml.tensorflow import TensorFlowModel
from autoai_agentrag.ml.pytorch import PyTorchModel

# Load a pre-trained model
model = MLModel.from_pretrained("pretrained-model-name")

# Or create a framework-specific model
tf_model = TensorFlowModel("tf-model")
tf_model.load("/path/to/tensorflow/model")
```

## Working with Agents

### Creating Agents

You can create different types of agents depending on your needs:

```python
from autoai_agentrag import Agent
from autoai_agentrag.agent.types import AgentType

# Task-oriented agent (default)
task_agent = Agent("task-agent", agent_type=AgentType.TASK_ORIENTED)

# Conversational agent
chat_agent = Agent("chat-agent", agent_type=AgentType.CONVERSATIONAL)

# Autonomous agent
auto_agent = Agent("auto-agent", agent_type=AgentType.AUTONOMOUS)
```

### Agent Configuration

Agents can be configured with various parameters:

```python
from autoai_agentrag.agent.types import AgentConfig

config = AgentConfig(
    max_retries=3,            # Maximum number of retry attempts for failed tasks
    timeout_seconds=60,       # Timeout for task execution
    allow_external_calls=True, # Whether the agent can make external API calls
    memory_size=10,           # Number of previous interactions to remember
    verbose=True              # Enable verbose logging
)

agent = Agent("configured-agent", config=config)
```

### Task Execution

Agents can execute tasks in different formats:

```python
# Execute a simple string task
result = agent.execute("Tell me about machine learning")

# Execute a structured task
task = {
    "query": "What is the weather today?",
    "location": "New York",
    "units": "metric"
}
result = agent.execute(task)

# Access the result
if result.success:
    print(f"Result: {result.result}")
else:
    print(f"Error: {result.error}")
```

## Working with RAG

### Setting Up a RAG Pipeline

```python
from autoai_agentrag import RAGPipeline

# Create a RAG pipeline
pipeline = RAGPipeline(name="my-pipeline")
```

### Adding Knowledge Sources

```python
# Add a source using a URL (the library will try to determine the source type)
pipeline.add_source("web", "https://api.example.com")

# Add a file-based source
pipeline.add_source("local", "/path/to/knowledge/base.json")

# Add a custom knowledge source
from autoai_agentrag.rag.sources import KnowledgeSource

class MyCustomSource(KnowledgeSource):
    def search(self, query, **kwargs):
        # Custom search implementation
        pass
        
    def get_document(self, doc_id):
        # Custom document retrieval
        pass

custom_source = MyCustomSource("custom")
pipeline.add_source("my-source", custom_source)
```

### Retrieving Information

```python
# Retrieve information from all sources
results = pipeline.retrieve("What is machine learning?", top_k=3)

# Process the results
for source_name, documents in results.items():
    print(f"Source: {source_name}")
    for doc in documents:
        print(f"- {doc.get('content', '')[:100]}...")
```

## Working with ML Models

### Loading Models

```python
from autoai_agentrag.ml.model import MLModel
from autoai_agentrag.ml.tensorflow import TensorFlowModel
from autoai_agentrag.ml.pytorch import PyTorchModel

# Load a model agnostically (library will infer the framework)
model = MLModel.from_pretrained("model-name-or-path")

# Load a specific framework model
tf_model = TensorFlowModel("tf-model")
tf_model.load("/path/to/tensorflow/model")

pt_model = PyTorchModel("pt-model")
pt_model.load("/path/to/pytorch/model.pt")
```

### Making Predictions

```python
# Prepare input data
import numpy as np
data = np.array([[1, 2, 3, 4, 5]])

# Make predictions
predictions = model.predict(data)
print(f"Predictions: {predictions}")

# Models can also be called directly
predictions = model(data)
```

### Framework-Specific Models

#### TensorFlow Models

```python
from autoai_agentrag.ml.tensorflow import TensorFlowModel, TensorFlowConfig

# Configure the model
config = TensorFlowConfig(
    eager_mode=True,            # Use eager execution
    mixed_precision=False,      # Disable mixed precision
    xla_compilation=True,       # Enable XLA compilation
    dynamic_shape=True          # Enable dynamic shapes
)

# Create and load the model
model = TensorFlowModel("tf-model", config=config)
model.load("/path/to/saved_model")

# Convert to TFLite
model.convert_to_tflite("/path/to/output.tflite", optimization="OPTIMIZE_FOR_SIZE")
```

#### PyTorch Models

```python
from autoai_agentrag.ml.pytorch import PyTorchModel, PyTorchConfig

# Configure the model
config = PyTorchConfig(
    use_jit=True,               # Use TorchScript (JIT)
    use_fp16=False,             # Disable half-precision
    device="cuda:0",            # Use specific GPU
    cudnn_benchmark=True        # Enable cuDNN benchmarking
)

# Create and load the model
model = PyTorchModel("pt-model", config=config)
model.load("/path/to/model.pt")
```

## Command-Line Interface

### Project Management

```bash
# Initialize a new project
autoai-agentrag init my-project

# Initialize with a specific template
autoai-agentrag init advanced-project --template full

# Initialize in a specific directory
autoai-agentrag init my-project --directory /path/to/projects
```

### Model Training

```bash
# Basic training
autoai-agentrag train --model classifier --data ./data/training

# Training with custom parameters
autoai-agentrag train \
  --model sentiment \
  --data ./data/reviews \
  --epochs 20 \
  --batch-size 64 \
  --output ./models
```

### Agent Deployment

```bash
# Deploy a basic agent
autoai-agentrag deploy --agent my-agent

# Deploy with custom settings
autoai-agentrag deploy \
  --agent my-agent \
  --port 9000 \
  --host 0.0.0.0 \
  --models-dir ./custom_models \
  --rag
```

## Advanced Topics

### Custom Agents

You can create custom agents by extending the base Agent class:

```python
from autoai_agentrag.agent.base import Agent as BaseAgent
from autoai_agentrag.agent.types import AgentType, TaskResult

class MyCustomAgent(BaseAgent):
    """A custom agent implementation

