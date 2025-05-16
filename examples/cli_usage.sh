#!/bin/bash
# =============================================================================
# AutoAI-AgentRAG - CLI Usage Examples
# =============================================================================
# This script demonstrates various ways to use the AutoAI-AgentRAG command-line
# interface (CLI) for different tasks and workflows.
#
# IMPORTANT: These are example commands meant to be copied and run individually,
# not to be executed as a complete script.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. PROJECT INITIALIZATION
# -----------------------------------------------------------------------------

# Display CLI version information
autoai-agentrag --version

# Show general help information
autoai-agentrag --help

# Initialize a new project with the default (basic) template
autoai-agentrag init my-project

# Initialize with the full template (includes Docker, notebooks, examples)
autoai-agentrag init ai-analytics-platform --template full

# Initialize with the minimal template (bare essentials only)
autoai-agentrag init quick-agent --template minimal

# Initialize in a specific directory
autoai-agentrag init customer-insights --directory ~/projects/ai

# Get help on the init command
autoai-agentrag init --help

# -----------------------------------------------------------------------------
# 2. MODEL TRAINING WORKFLOWS
# -----------------------------------------------------------------------------

# Train a basic classification model using default parameters
autoai-agentrag train --model text-classification --data ./data/texts

# Train with custom parameters
autoai-agentrag train \
  --model sentiment-analysis \
  --data ./data/reviews \
  --epochs 20 \
  --batch-size 64 \
  --output ./custom_models

# Train using a configuration file
autoai-agentrag train \
  --model entity-recognition \
  --data ./data/documents \
  --config ./configs/training_config.yaml

# Get help on the train command
autoai-agentrag train --help

# -----------------------------------------------------------------------------
# 3. AGENT DEPLOYMENT SCENARIOS
# -----------------------------------------------------------------------------

# Deploy a basic agent with default settings (localhost:8000)
autoai-agentrag deploy --agent my-agent

# Deploy with custom port and host
autoai-agentrag deploy \
  --agent customer-service-agent \
  --port 9000 \
  --host 0.0.0.0

# Deploy with RAG capabilities enabled
autoai-agentrag deploy \
  --agent research-assistant \
  --rag

# Deploy with custom model directory
autoai-agentrag deploy \
  --agent data-analyzer \
  --models-dir ./trained_models

# Deploy with specific configuration
autoai-agentrag deploy \
  --agent production-agent \
  --config ./configs/deployment_config.yaml

# Get help on the deploy command
autoai-agentrag deploy --help

# -----------------------------------------------------------------------------
# 4. PLUGIN INSTALLATION AND MANAGEMENT
# -----------------------------------------------------------------------------

# Install a plugin from a local file
autoai-agentrag install-plugin --plugin-path ./plugins/custom_source.py

# Install a plugin with a custom name
autoai-agentrag install-plugin \
  --plugin-path ./plugins/database_connector \
  --name db-connector

# Get help on plugin installation
autoai-agentrag install-plugin --help

# -----------------------------------------------------------------------------
# 5. COMMON USAGE PATTERNS AND BEST PRACTICES
# -----------------------------------------------------------------------------

# Typical workflow example 1: Initialize, train, and deploy
# Step 1: Initialize a project
autoai-agentrag init customer-insights --template full

# Step 2: Navigate to the project directory
cd customer-insights

# Step 3: Train a model using project data
autoai-agentrag train --model customer-classifier --data ./data/customers

# Step 4: Deploy the agent with the trained model
autoai-agentrag deploy --agent customer-insights-agent --rag

# Typical workflow example 2: Verbose mode for debugging
# Run any command with verbose flag for detailed logging
autoai-agentrag --verbose deploy --agent debug-agent

# Typical workflow example 3: Using environment variables for configuration
# Export variables before running commands
export AUTOAI_LOG_LEVEL=DEBUG
export AUTOAI_MODELS_DIR=~/models
autoai-agentrag deploy --agent env-configured-agent

# -----------------------------------------------------------------------------
# 6. ADVANCED USE CASES
# -----------------------------------------------------------------------------

# Chaining commands with output redirection
autoai-agentrag train --model complex-model --data ./big_dataset > training_log.txt

# Running multiple instances on different ports
autoai-agentrag deploy --agent agent1 --port 8000 &
autoai-agentrag deploy --agent agent2 --port 8001 &

# Monitoring agent logs
autoai-agentrag deploy --agent logging-agent --log-file agent.log

# Scheduled retraining (using cron syntax)
# Add to crontab: 0 2 * * * cd /path/to/project && autoai-agentrag train --model daily-model --data ./data

