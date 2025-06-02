"""
Model Context Protocol (MCP) Integration Package

This package provides integration between analystt1 and the Model Context Protocol (MCP),
an open standard for connecting AI models to external tools and data sources.

The MCP module enables:
- Standardized tool interfaces for CrewAI agents
- Dynamic tool discovery and execution
- Lifecycle management for MCP servers
- Integration with Gemini 2.5 native MCP support

Usage:
    from backend.mcp import MCPClient, get_mcp_client
    
    # Using the client directly
    client = MCPClient()
    tools = client.get_tools("graph")
    
    # Using the factory function
    client = get_mcp_client()
    
    # Context manager pattern
    with MCPClient() as client:
        tools = client.get_all_tools()
        # Use tools in CrewAI agents
    
    # Getting tools for CrewFactory
    from backend.mcp import get_mcp_tools_for_factory
    mcp_tools = get_mcp_tools_for_factory()
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Set

# Re-export key classes and functions
from backend.mcp.client import (
    MCPClient,
    get_mcp_client,
    get_mcp_tools_for_factory,
)

# Constants
MCP_ENABLED = os.environ.get("ENABLE_MCP", "0") == "1"
MCP_MODE = os.environ.get("MCP_MODE", "development")  # 'development' or 'production'
DEFAULT_REGISTRY_PATH = os.environ.get(
    "MCP_REGISTRY_PATH", 
    Path(__file__).parent.parent.parent / "config" / "mcp_servers.yaml"
)

# Version
__version__ = "0.1.0"

__all__ = [
    "MCPClient",
    "get_mcp_client",
    "get_mcp_tools_for_factory",
    "MCP_ENABLED",
    "MCP_MODE",
    "DEFAULT_REGISTRY_PATH",
]
