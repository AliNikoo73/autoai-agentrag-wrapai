#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoAI-AgentRAG documentation build configuration file.

This file is executed by Sphinx to build the documentation.
"""

import os
import sys
import datetime

# Add the autoai-agentrag package to the path
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'AutoAI-AgentRAG'
copyright = f'{datetime.datetime.now().year}, AutoAI-AgentRAG Team'
author = 'AutoAI-AgentRAG Team'

# Try to get the version from the package
try:
    from autoai_agentrag import __version__
    version = __version__
    release = __version__
except ImportError:
    version = '0.1.0'
    release = '0.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings
extensions = [
    'sphinx.ext.autodoc',  # Automatically generate API documentation
    'sphinx.ext.viewcode',  # Add links to the source code
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx.ext.coverage',  # Checks for documentation coverage
    'sphinx.ext.autosummary',  # Generate summary tables for API docs
    'sphinx.ext.todo',  # Support for TODO items
    'sphinx.ext.mathjax',  # Render math via MathJax
    'sphinx_rtd_theme',  # ReadTheDocs theme
    'recommonmark',  # Markdown support
    'myst_parser',  # Enhanced Markdown support
]

# Auto-generate API docs
autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Markdown support
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The main toctree document
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom sidebar templates, maps document names to template names
html_sidebars = {
    '**': [
        'relations.html',  # navigation links
        'searchbox.html',  # search box
        'globaltoc.html',  # global table of contents
    ]
}

# -- Options for Intersphinx -------------------------------------------------
# References to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'tensorflow': ('https://www.tensorflow.org/api_docs/python', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

# -- MyST Markdown settings --------------------------------------------------
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'strikethrough',
    'substitution',
    'tasklist',
]

# -- ReadTheDocs configuration -----------------------------------------------
# Build docs in 'html' format by default
html_baseurl = 'https://autoai-agentrag.readthedocs.io/'

# Tell sphinx what the primary language being documented is
language = 'en'

# Tell sphinx what the pygments highlight lexer should use
highlight_language = 'python3'

# If true, "(C) Copyright ..." is shown in the HTML footer
html_show_copyright = True

# -- Autodoc configuration --------------------------------------------------
# Ensure that constructors are documented
autoclass_content = 'both'

# Only show class docstring and init method docstring, not other class methods
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'undoc-members': True,
    'inherited-members': False,
    'special-members': '__init__',
}

# Mock imports that may not be available during doc build
autodoc_mock_imports = [
    'tensorflow',
    'torch',
    'numpy',
    'pandas',
    'sklearn',
    'transformers',
    'faiss',
]

# -- Options for HTML help output -------------------------------------------
htmlhelp_basename = 'AutoAIAgentRAGdoc'

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'AutoAIAgentRAG.tex', 'AutoAI-AgentRAG Documentation',
     'AutoAI-AgentRAG Team', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, 'autoaiagentrag', 'AutoAI-AgentRAG Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (master_doc, 'AutoAIAgentRAG', 'AutoAI-AgentRAG Documentation',
     author, 'AutoAIAgentRAG', 'AI Agent framework with RAG and ML capabilities.',
     'Miscellaneous'),
]

# -- Custom setup for Read the Docs ------------------------------------------
def setup(app):
    """Setup function to customize sphinx behavior."""
    # To add custom CSS
    # app.add_css_file('custom.css')
    # To add custom JavaScript
    # app.add_js_file('custom.js')
    pass

