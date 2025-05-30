"""
CrewAI-based Multi-Agent System for Analyst Augmentation.

This package implements a multi-agent system using CrewAI to enhance
analytical capabilities for financial crime detection and investigation.

The system consists of specialized agents that work together to:
- Convert natural language to Cypher queries
- Execute graph database operations
- Detect fraud patterns
- Run secure code in sandboxed environments
- Generate comprehensive reports

Each agent has specific roles, goals, and tools to accomplish tasks
within a coordinated workflow.
"""

from backend.agents.config import AgentConfig
from backend.agents.factory import CrewFactory
from backend.agents.tools import (
    GraphQueryTool,
    SandboxExecTool,
    CodeGenTool,
)
from backend.agents.llm import GeminiLLMProvider

# Version
__version__ = "0.1.0"

# Export commonly used components
__all__ = [
    "AgentConfig",
    "CrewFactory",
    "GraphQueryTool",
    "SandboxExecTool",
    "CodeGenTool",
    "GeminiLLMProvider",
]
