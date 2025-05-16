#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Agent Example

This example demonstrates the basic usage of the Agent class from AutoAI-AgentRAG.
It shows how to create, configure and run an agent for simple tasks.
"""

import logging
import sys
from typing import Dict, Any

from autoai_agentrag import Agent
from autoai_agentrag.agent.types import AgentType, AgentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    """
    Demonstrate basic Agent usage.
    
    This function shows how to:
    1. Create an agent with different configurations
    2. Execute tasks with the agent
    3. Handle results and errors
    """
    try:
        # Create a simple agent with default configuration
        logger.info("Creating a basic agent with default configuration")
        basic_agent = Agent(name="basic-agent")
        logger.info(f"Agent created: {basic_agent}")
        
        # Execute a simple task
        logger.info("Executing a simple task with the basic agent")
        result = basic_agent.execute("What is the capital of France?")
        
        # Display the result
        if result.success:
            logger.info(f"Task completed successfully: {result.result}")
        else:
            logger.error(f"Task failed: {result.error}")
        
        # Create an agent with custom configuration
        logger.info("\nCreating an agent with custom configuration")
        custom_config = AgentConfig(
            max_retries=5,
            timeout_seconds=120,
            allow_external_calls=True,
            memory_size=20,
            verbose=True
        )
        conversational_agent = Agent(
            name="conversational-agent",
            agent_type=AgentType.CONVERSATIONAL,
            config=custom_config
        )
        logger.info(f"Agent created: {conversational_agent}")
        
        # Execute a more complex task
        logger.info("Executing a conversational task")
        task_input = {
            "query": "Tell me about machine learning",
            "context": "The user is a beginner in AI",
            "max_length": 100
        }
        result = conversational_agent.execute(task_input)
        
        # Display the result
        if result.success:
            logger.info(f"Task completed successfully: {result.result}")
        else:
            logger.error(f"Task failed: {result.error}")
            
        # Demonstrate error handling
        try:
            # Intentionally cause an error
            logger.info("\nDemonstrating error handling with an invalid task")
            result = basic_agent.execute(None)  # Invalid input
        except Exception as e:
            logger.error(f"Caught exception: {str(e)}")
        
        logger.info("Basic agent demonstration completed")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

