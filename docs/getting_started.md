# Getting Started with AutoAI-AgentRAG

This guide will help you get up and running with AutoAI-AgentRAG quickly. We'll cover installation, basic setup, and a simple "Hello World" example.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Basic Installation

Install AutoAI-AgentRAG using pip:

```bash
pip install autoai-agentrag
```

For development or to get the latest features:

```bash
pip install git+https://github.com/autoai-agentrag/autoai-agentrag.git
```

### Optional Dependencies

AutoAI-AgentRAG has several optional dependencies for different features:

- TensorFlow support:
  ```bash
  pip install autoai-agentrag[tensorflow]
  ```

- PyTorch support:
  ```bash
  pip install autoai-agentrag[pytorch]
  ```

- All optional dependencies:
  ```bash
  pip install autoai-agentrag[all]
  ```

## Project Initialization

The easiest way to get started is to use the CLI to create a new project:

```bash
# Create a new project with basic template
autoai-agentrag init my-first-project

# Change into project directory
cd my-first-project
```

This creates a new project with the following structure:

```
my-first-project/
├── agents/          # Custom agent implementations
├── configs/         # Configuration files
│   └── config.yaml
├── data/            # Data directory for models
├── models/          # Saved models
├── README.md        # Project documentation
├── requirements.txt # Project dependencies
└── run.py           # Example script
```

## Quick Start Example

Let's create a simple agent that can answer questions:

```python
from autoai_agentrag import Agent, RAGPipeline

# Create a simple agent
agent = Agent("my-first-agent")

# Create a RAG pipeline for knowledge retrieval
rag = RAGPipeline()

# Add a knowledge source (this example uses a mock URL)
rag.add_source("web", "https://example.com/api")

# Connect the RAG pipeline to the agent
agent.connect_rag(rag)

# Execute a task
result = agent.execute("What is the capital of France?")

# Print the result
print(f"Success: {result.success}")
print(f"Result: {result.result}")
```

Save this as `hello_world.py` in your project directory and run it:

```bash
python hello_world.py
```

## Using the CLI

AutoAI-AgentRAG comes with a command-line interface for common operations:

```bash
# Show available commands
autoai-agentrag --help

# Train a model
autoai-agentrag train --model text-classification --data ./data/texts

# Deploy an agent
autoai-agentrag deploy --agent my-agent --port 8000
```

## Next Steps

Now that you have a basic understanding of AutoAI-AgentRAG, you can:

- Read the [User Guide](user_guide.md) for more detailed information
- Explore the [Examples](examples.md) to see what's possible
- Check the [API Reference](api_reference.md) for detailed API documentation
- Learn how to create custom agents in the [Advanced Topics](user_guide.md#advanced-topics) section

