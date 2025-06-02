"""
MCP (Model Context Protocol) Servers Module

This package contains MCP server implementations for analystt1, providing standardized
interfaces for AI models to interact with tools, data sources, and prompts.

MCP servers expose functionality through a JSON-RPC 2.0 interface, allowing
any MCP-compatible client (including Gemini 2.5, Claude, etc.) to discover and
use the tools defined here.

Usage:
    from backend.mcp_servers import registry, get_server_config
    
    # Get all registered server configs
    all_servers = registry.get_all_servers()
    
    # Get config for a specific server
    graph_config = get_server_config("graph")
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

# Default path for MCP server registry
DEFAULT_REGISTRY_PATH = os.environ.get(
    "MCP_REGISTRY_PATH", 
    Path(__file__).parent.parent.parent / "config" / "mcp_servers.yaml"
)

class MCPServerRegistry:
    """Registry for MCP servers in the analystt1 system."""
    
    def __init__(self, registry_path: Optional[Union[str, Path]] = None):
        """
        Initialize the MCP server registry.
        
        Args:
            registry_path: Path to the YAML registry file. If None, uses DEFAULT_REGISTRY_PATH.
        """
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self._servers: Dict[str, Dict] = {}
        self._loaded = False
    
    def load(self) -> None:
        """Load server configurations from the registry file."""
        if not self.registry_path.exists():
            logger.warning(f"MCP registry file not found: {self.registry_path}")
            return
        
        try:
            import yaml
            with open(self.registry_path, 'r') as f:
                self._servers = yaml.safe_load(f) or {}
            self._loaded = True
            logger.info(f"Loaded {len(self._servers)} MCP servers from {self.registry_path}")
        except Exception as e:
            logger.error(f"Error loading MCP registry: {e}")
            self._servers = {}
    
    def get_all_servers(self) -> Dict[str, Dict]:
        """
        Get all registered server configurations.
        
        Returns:
            Dictionary of server configurations keyed by server name.
        """
        if not self._loaded:
            self.load()
        return self._servers
    
    def get_server(self, name: str) -> Optional[Dict]:
        """
        Get configuration for a specific server.
        
        Args:
            name: Name of the server.
            
        Returns:
            Server configuration or None if not found.
        """
        if not self._loaded:
            self.load()
        return self._servers.get(name)
    
    def register_server(self, name: str, config: Dict) -> None:
        """
        Register a new server configuration.
        
        Args:
            name: Name of the server.
            config: Server configuration.
        """
        if not self._loaded:
            self.load()
        self._servers[name] = config
        logger.info(f"Registered MCP server: {name}")


# Create global registry instance
registry = MCPServerRegistry()

def get_server_config(name: str) -> Optional[Dict]:
    """
    Get configuration for a specific MCP server.
    
    Args:
        name: Name of the server.
        
    Returns:
        Server configuration or None if not found.
    """
    return registry.get_server(name)


__all__ = ["registry", "get_server_config", "MCPServerRegistry"]
