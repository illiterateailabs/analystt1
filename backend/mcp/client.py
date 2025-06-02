"""
MCP Client Integration Module

This module provides a clean interface for CrewAI agents to use Model Context Protocol (MCP)
servers. It wraps the crewai-mcp-toolbox package and handles server lifecycle management,
tool discovery, and execution.

Usage:
    from backend.mcp.client import MCPClient
    
    # Initialize the client
    mcp_client = MCPClient()
    
    # Get tools for a specific server
    graph_tools = mcp_client.get_tools("graph")
    
    # Get all available tools
    all_tools = mcp_client.get_all_tools()
    
    # Use in a context manager to ensure proper cleanup
    with MCPClient() as client:
        tools = client.get_all_tools()
        # Use tools in CrewAI agents
"""

import os
import logging
import yaml
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
from contextlib import contextmanager

from crewai import BaseTool
from crewai_mcp_toolbox import MCPToolSet

from backend.core.logging import get_logger

# Configure logger
logger = get_logger(__name__)

# Default paths
DEFAULT_REGISTRY_PATH = os.environ.get(
    "MCP_REGISTRY_PATH", 
    Path(__file__).parent.parent.parent / "config" / "mcp_servers.yaml"
)

# Environment variables
MCP_ENABLED = os.environ.get("ENABLE_MCP", "0") == "1"
MCP_MODE = os.environ.get("MCP_MODE", "development")  # 'development' or 'production'


class MCPClient:
    """
    Client for Model Context Protocol (MCP) servers.
    
    This class provides a clean interface for CrewAI agents to use MCP servers.
    It handles server lifecycle management, tool discovery, and execution.
    """
    
    def __init__(
        self,
        registry_path: Optional[Union[str, Path]] = None,
        mode: str = MCP_MODE,
        enabled: bool = MCP_ENABLED
    ):
        """
        Initialize the MCP client.
        
        Args:
            registry_path: Path to the YAML registry file. If None, uses DEFAULT_REGISTRY_PATH.
            mode: 'development' or 'production'. Development uses stdio, production uses HTTP.
            enabled: Whether MCP is enabled. If False, get_tools() returns empty lists.
        """
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self.mode = mode
        self.enabled = enabled
        self._registry: Dict[str, Dict] = {}
        self._toolsets: Dict[str, MCPToolSet] = {}
        self._active_servers: Set[str] = set()
        
        # Load the registry on initialization
        if self.enabled:
            self._load_registry()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def _load_registry(self) -> None:
        """Load server configurations from the registry file."""
        if not self.registry_path.exists():
            logger.warning(f"MCP registry file not found: {self.registry_path}")
            return
        
        try:
            with open(self.registry_path, 'r') as f:
                self._registry = yaml.safe_load(f) or {}
            
            # Filter out disabled servers
            self._registry = {
                name: config for name, config in self._registry.items()
                if config.get("enabled", True)
            }
            
            logger.info(f"Loaded {len(self._registry)} MCP servers from {self.registry_path}")
        except Exception as e:
            logger.error(f"Error loading MCP registry: {e}")
            self._registry = {}
    
    def get_server_config(self, server_name: str) -> Optional[Dict]:
        """
        Get configuration for a specific server.
        
        Args:
            server_name: Name of the server.
            
        Returns:
            Server configuration or None if not found.
        """
        return self._registry.get(server_name)
    
    def get_all_server_configs(self) -> Dict[str, Dict]:
        """
        Get all server configurations.
        
        Returns:
            Dictionary of server configurations keyed by server name.
        """
        return self._registry
    
    def start_server(self, server_name: str) -> Optional[MCPToolSet]:
        """
        Start an MCP server and return its toolset.
        
        Args:
            server_name: Name of the server to start.
            
        Returns:
            MCPToolSet for the server or None if the server is not found or fails to start.
        """
        if not self.enabled:
            logger.info("MCP is disabled. Set ENABLE_MCP=1 to enable.")
            return None
        
        # Check if server is already running
        if server_name in self._toolsets:
            logger.info(f"MCP server '{server_name}' is already running")
            return self._toolsets[server_name]
        
        # Get server configuration
        server_config = self.get_server_config(server_name)
        if not server_config:
            logger.warning(f"MCP server '{server_name}' not found in registry")
            return None
        
        try:
            # Create a temporary config file for this server
            temp_config = {server_name: server_config}
            temp_config_path = Path(f"/tmp/mcp_{server_name}.yaml")
            
            # Handle environment variables in the config
            if "env" in server_config:
                for key, value in server_config["env"].items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        env_var = value[2:-1]
                        # Handle default values with :- syntax
                        if ":-" in env_var:
                            env_name, default = env_var.split(":-", 1)
                            server_config["env"][key] = os.environ.get(env_name, default)
                        else:
                            server_config["env"][key] = os.environ.get(env_var, "")
            
            # Write the temporary config
            with open(temp_config_path, 'w') as f:
                yaml.dump(temp_config, f)
            
            # Create the toolset
            logger.info(f"Starting MCP server '{server_name}'")
            toolset = MCPToolSet(registry_path=str(temp_config_path))
            
            # Store the toolset
            self._toolsets[server_name] = toolset
            self._active_servers.add(server_name)
            
            # Clean up the temporary config
            temp_config_path.unlink(missing_ok=True)
            
            return toolset
        except Exception as e:
            logger.error(f"Error starting MCP server '{server_name}': {e}")
            return None
    
    def stop_server(self, server_name: str) -> bool:
        """
        Stop an MCP server.
        
        Args:
            server_name: Name of the server to stop.
            
        Returns:
            True if the server was stopped successfully, False otherwise.
        """
        if server_name not in self._toolsets:
            logger.warning(f"MCP server '{server_name}' is not running")
            return False
        
        try:
            # Clean up the toolset
            self._toolsets[server_name].cleanup()
            del self._toolsets[server_name]
            self._active_servers.remove(server_name)
            logger.info(f"Stopped MCP server '{server_name}'")
            return True
        except Exception as e:
            logger.error(f"Error stopping MCP server '{server_name}': {e}")
            return False
    
    def get_tools(self, server_name: str) -> List[BaseTool]:
        """
        Get tools for a specific server.
        
        Args:
            server_name: Name of the server.
            
        Returns:
            List of CrewAI BaseTool instances for the server.
        """
        if not self.enabled:
            return []
        
        # Start the server if it's not already running
        if server_name not in self._toolsets:
            toolset = self.start_server(server_name)
            if not toolset:
                return []
        
        # Get the tools from the toolset
        try:
            tools = list(self._toolsets[server_name])
            logger.info(f"Got {len(tools)} tools from MCP server '{server_name}'")
            return tools
        except Exception as e:
            logger.error(f"Error getting tools from MCP server '{server_name}': {e}")
            return []
    
    def get_all_tools(self) -> List[BaseTool]:
        """
        Get tools from all registered servers.
        
        Returns:
            List of CrewAI BaseTool instances from all servers.
        """
        if not self.enabled:
            return []
        
        all_tools = []
        for server_name in self._registry:
            tools = self.get_tools(server_name)
            all_tools.extend(tools)
        
        logger.info(f"Got {len(all_tools)} tools from all MCP servers")
        return all_tools
    
    def cleanup(self) -> None:
        """Clean up all running servers."""
        for server_name in list(self._active_servers):
            self.stop_server(server_name)
        
        # Clear the toolsets
        self._toolsets.clear()
        self._active_servers.clear()
        logger.info("Cleaned up all MCP servers")
    
    @contextmanager
    def server_session(self, server_name: str):
        """
        Context manager for a specific server session.
        
        Args:
            server_name: Name of the server.
            
        Yields:
            List of CrewAI BaseTool instances for the server.
        """
        toolset = self.start_server(server_name)
        try:
            yield list(toolset) if toolset else []
        finally:
            self.stop_server(server_name)
    
    @property
    def is_enabled(self) -> bool:
        """Check if MCP is enabled."""
        return self.enabled
    
    @property
    def active_servers(self) -> List[str]:
        """Get the list of active server names."""
        return list(self._active_servers)


# Factory function to get an MCPClient instance
def get_mcp_client(
    registry_path: Optional[Union[str, Path]] = None,
    mode: str = MCP_MODE,
    enabled: bool = MCP_ENABLED
) -> MCPClient:
    """
    Get an MCPClient instance.
    
    Args:
        registry_path: Path to the YAML registry file. If None, uses DEFAULT_REGISTRY_PATH.
        mode: 'development' or 'production'. Development uses stdio, production uses HTTP.
        enabled: Whether MCP is enabled. If False, get_tools() returns empty lists.
        
    Returns:
        MCPClient instance.
    """
    return MCPClient(registry_path=registry_path, mode=mode, enabled=enabled)


# Integration with CrewAI factory pattern
def get_mcp_tools_for_factory() -> List[BaseTool]:
    """
    Get all MCP tools for the CrewAI factory.
    
    This function is designed to be called from the CrewFactory to get all
    available MCP tools.
    
    Returns:
        List of CrewAI BaseTool instances from all MCP servers.
    """
    if not MCP_ENABLED:
        return []
    
    try:
        with MCPClient() as client:
            return client.get_all_tools()
    except Exception as e:
        logger.error(f"Error getting MCP tools for factory: {e}")
        return []
