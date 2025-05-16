# Contributing to AutoAI-AgentRAG

Thank you for your interest in contributing to AutoAI-AgentRAG! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Coding Standards and Guidelines](#coding-standards-and-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Requirements](#testing-requirements)
- [Documentation Requirements](#documentation-requirements)
- [Issue Reporting Guidelines](#issue-reporting-guidelines)
- [Community and Communication](#community-and-communication)

## Setting Up the Development Environment

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda (for package management)

### Installation Steps

1. Fork the repository on GitHub

2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/autoai-agentrag-wrapai.git
   cd autoai-agentrag-wrapai
   ```

3. Set up a virtual environment:
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n autoai-env python=3.9
   conda activate autoai-env
   ```

4. Install the package in development mode with all extra dependencies:
   ```bash
   pip install -e ".[dev,test,docs]"
   ```

5. Set up pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Coding Standards and Guidelines

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code style
- Use [Black](https://black.readthedocs.io/) for code formatting
- Sort imports using [isort](https://pycqa.github.io/isort/)
- Maximum line length of 88 characters (Black default)

### Docstrings

- Use [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Document all public classes, methods, and functions
- Include type hints in the function signatures

Example:
```python
def function_name(param1: str, param2: int) -> bool:
    """
    Short description of function.
    
    Longer description explaining the function's purpose and behavior.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is invalid
    """
    # Function body
```

### Commit Messages

- Use clear, descriptive commit messages
- Follow the conventional commits format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `style:` for formatting changes
  - `refactor:` for code refactoring
  - `test:` for adding or modifying tests
  - `chore:` for maintenance tasks

## Pull Request Process

1. **Branch naming**: Use a descriptive branch name that reflects the changes you're making, e.g., `feature/add-new-agent-type` or `fix/rag-pipeline-error`

2. **Keep PRs focused**: Each PR should address a single concern or feature

3. **PR Template**: Fill out the PR template completely, including:
   - Description of changes
   - Related issue numbers
   - Screenshots or examples (if applicable)
   - Checklist of completed tasks

4. **Code Review Process**:
   - All PRs require at least one review from a maintainer
   - Address all review comments
   - Make sure all CI checks pass

5. **Merge Requirements**:
   - PR must be approved by at least one maintainer
   - All discussions must be resolved
   - CI pipelines must pass (tests, linting, etc.)
   - Documentation must be updated if necessary

## Testing Requirements

### Writing Tests

- Write unit tests for all new code
- Use [pytest](https://docs.pytest.org/) for writing and running tests
- Aim for at least 80% code coverage for new features
- Include both positive and negative test cases

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=autoai_agentrag

# Run specific test file
pytest tests/test_specific_module.py
```

### Mocking External Dependencies

- Use `unittest.mock` or `pytest-mock` for mocking external dependencies
- Don't rely on external services in unit tests
- Use fixtures for common test setups

## Documentation Requirements

### API Documentation

- Document all public APIs
- Keep docstrings up-to-date with code changes
- Include examples in docstrings where helpful

### User Documentation

- Update relevant user documentation for new features or changes
- Add examples to demonstrate new functionality
- Ensure documentation builds correctly

### Building Documentation Locally

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

Then open `_build/html/index.html` in your browser to review.

## Issue Reporting Guidelines

### Bug Reports

When reporting bugs, please include:

1. A clear and descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment information (OS, Python version, package version)
6. Any relevant logs or error messages

### Feature Requests

When requesting new features, please include:

1. A clear and descriptive title
2. Description of the problem the feature would solve
3. Proposed solution or implementation ideas
4. Any additional context or examples

## Community and Communication

- Join our [Discord server](https://discord.gg/autoai-agentrag) for real-time discussions
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
- Be respectful and inclusive in all communications

Thank you for contributing to AutoAI-AgentRAG!

