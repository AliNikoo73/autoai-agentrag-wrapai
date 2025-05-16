#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="autoai-agentrag",
    version="0.1.0",
    author="AutoAI-AgentRAG Team",
    author_email="info@autoai-agentrag.com",
    description="An intelligent automation library integrating AI Agents, RAG, and ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autoai-agentrag/autoai-agentrag",
    packages=find_packages(include=["autoai_agentrag", "autoai_agentrag.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "flake8>=3.8.0",
            "black>=20.8b1",
            "isort>=5.0.0",
            "sphinx>=3.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "tensorflow": ["tensorflow>=2.4.0"],
        "pytorch": ["torch>=1.7.0", "torchvision>=0.8.0"],
    },
    entry_points={
        "console_scripts": [
            "autoai-agentrag=autoai_agentrag.cli.main:main",
        ],
    },
)

