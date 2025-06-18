"""
Jobs Package for Analyst Augmentation Agent

This package contains background job implementations for processing and analyzing
blockchain data. Jobs are designed to run asynchronously and can be scheduled
or triggered manually.

Available job types:
- Graph ingestion jobs (Sim API data → Neo4j)
- Blockchain data analysis jobs
- Scheduled data refresh jobs
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable

# Set up package-level logger
logger = logging.getLogger(__name__)

# Package version
__version__ = "0.1.0"

# Export key classes and functions
# These will be populated as job modules are added
__all__ = []
