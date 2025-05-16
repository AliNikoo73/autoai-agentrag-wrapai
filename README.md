# AutoAI-AgentRAG

[![Tests](https://github.com/AliNikoo73/autoai-agentrag-wrapai/actions/workflows/tests.yml/badge.svg)](https://github.com/AliNikoo73/autoai-agentrag-wrapai/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/autoai-agentrag-wrapai/badge/?version=latest)](https://autoai-agentrag-wrapai.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://img.shields.io/pypi/v/autoai-agentrag.svg)](https://pypi.org/project/autoai-agentrag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/autoai-agentrag.svg)](https://pypi.org/project/autoai-agentrag/)

AutoAI-AgentRAG is an open-source PyPI library designed to enhance automation workflows by integrating AI Agents, Retrieval-Augmented Generation (RAG), and Machine Learning (ML). It empowers developers to build intelligent automation systems that efficiently manage complex tasks with minimal manual intervention.

## Features

### AI Agent Framework
A modular system for designing and managing AI Agents that autonomously execute tasks and make context-informed decisions.

### Retrieval-Augmented Generation (RAG) Integration
Connects to external knowledge bases (APIs, databases, web sources) for real-time data retrieval, enhancing contextual awareness.

### Machine Learning Models
Offers pre-trained and customizable ML models compatible with TensorFlow, PyTorch, and other frameworks for predictive analytics and pattern recognition.

### Command-Line Interface (CLI)
A user-friendly CLI for initializing projects, training models, and deploying agents, facilitating ease of use for developers.

### Extensible Plugin System
Enables users to add custom data sources, ML models, or agent behaviors to adapt the library to specific use cases.

### Comprehensive Documentation
Detailed guides, API references, and tutorials hosted on ReadTheDocs to accelerate user onboarding and development.

### Unit Testing and CI/CD
Built-in unit tests and GitHub Actions for continuous integration and deployment, ensuring code reliability and maintainability.

### Cross-Platform Compatibility
Optimized for Windows, macOS, and Linux, with Docker support for flexible deployment across environments.

### Real-Time Monitoring Dashboard
An optional web interface to track agent performance, task progress, and ML model metrics for better visibility and control.

### Community Support
A Discord server and GitHub Issues page for collaboration, troubleshooting, and feature requests to foster an active user community.

## Installation

```bash
pip install autoai-agentrag
```

For development version:

```bash
pip install git+https://github.com/AliNikoo73/autoai-agentrag-wrapai.git
```

## Quick Start

```python
from autoai_agentrag import Agent, RAGPipeline, MLModel

# Create an AI agent with RAG capabilities
agent = Agent("my-agent")
rag = RAGPipeline()
rag.add_source("web_api", "https://api.example.com/knowledge")
agent.connect_rag(rag)

# Load a ML model
model = MLModel.from_pretrained("classification_model")
agent.add_model(model)

# Execute tasks
result = agent.execute("Analyze this data and provide insights")
print(result)
```

## CLI Usage

```bash
# Initialize a new project
autoai-agentrag init my-automation-project

# Train a model
autoai-agentrag train --model classification --data path/to/data

# Deploy agent
autoai-agentrag deploy --agent my-agent --port 8000
```

## Documentation

For detailed documentation, visit [ReadTheDocs](https://autoai-agentrag-wrapai.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

