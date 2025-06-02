"""
Tests for Model Context Protocol (MCP) integration in analystt1.

This module tests the MCP client, server discovery, tool execution,
and integration with Gemini. It verifies that MCP servers can be
started, tools can be discovered, and the integration with CrewAI
and Gemini works correctly.
"""

import os
import json
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import yaml

from crewai import BaseTool
from crewai_mcp_toolbox import MCPToolSet
from mcpengine import Server, Tool, Context

from backend.mcp.client import MCPClient, get_mcp_client, get_mcp_tools_for_factory
from backend.mcp_servers.echo_server import server as echo_server
from backend.agents.tools import get_all_tools


# Fixtures
@pytest.fixture
def temp_registry_file():
    """Create a temporary registry file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        registry_content = {
            "test_echo": {
                "command": "python",
                "args": ["backend/mcp_servers/echo_server.py"],
                "transport": "stdio",
                "enabled": True,
                "description": "Test echo server"
            }
        }
        yaml.dump(registry_content, f)
        f.flush()
        yield Path(f.name)
    
    # Cleanup
    try:
        os.unlink(f.name)
    except Exception:
        pass


@pytest.fixture
def mock_toolset():
    """Mock MCPToolSet for testing."""
    mock = MagicMock(spec=MCPToolSet)
    
    # Create a mock tool
    mock_tool = MagicMock(spec=BaseTool)
    mock_tool.name = "echo"
    mock_tool.description = "Echo tool for testing"
    
    # Make the toolset iterable and return the mock tool
    mock.__iter__.return_value = [mock_tool]
    
    return mock


@pytest.fixture
def mock_server():
    """Mock MCP server for testing."""
    server = MagicMock(spec=Server)
    server.name = "test-server"
    
    # Mock tool registration
    @server.tool(
        name="test_tool",
        description="Test tool",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}}
    )
    async def test_tool(ctx, text):
        return {"result": f"Test: {text}"}
    
    return server


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for testing."""
    mock = MagicMock()
    mock.aio = MagicMock()
    mock.aio.models = MagicMock()
    mock.aio.models.generate_content = AsyncMock()
    mock.aio.models.generate_content.return_value = MagicMock(text="Test response")
    return mock


# Unit Tests
class TestMCPClient:
    """Tests for the MCPClient class."""
    
    def test_init(self, temp_registry_file):
        """Test MCPClient initialization."""
        client = MCPClient(registry_path=temp_registry_file, enabled=True)
        assert client.registry_path == temp_registry_file
        assert client.enabled is True
        assert client.mode == "development"
        assert len(client._registry) == 1
        assert "test_echo" in client._registry
    
    def test_init_disabled(self, temp_registry_file):
        """Test MCPClient initialization when disabled."""
        client = MCPClient(registry_path=temp_registry_file, enabled=False)
        assert client.enabled is False
        assert len(client._registry) == 0  # Registry not loaded when disabled
    
    def test_get_server_config(self, temp_registry_file):
        """Test getting server configuration."""
        client = MCPClient(registry_path=temp_registry_file, enabled=True)
        config = client.get_server_config("test_echo")
        assert config is not None
        assert config["command"] == "python"
        assert config["transport"] == "stdio"
    
    def test_get_all_server_configs(self, temp_registry_file):
        """Test getting all server configurations."""
        client = MCPClient(registry_path=temp_registry_file, enabled=True)
        configs = client.get_all_server_configs()
        assert len(configs) == 1
        assert "test_echo" in configs
    
    @patch("backend.mcp.client.MCPToolSet")
    def test_start_server(self, mock_toolset_class, temp_registry_file):
        """Test starting an MCP server."""
        mock_toolset_class.return_value = MagicMock()
        
        client = MCPClient(registry_path=temp_registry_file, enabled=True)
        result = client.start_server("test_echo")
        
        assert result is not None
        assert "test_echo" in client._toolsets
        assert "test_echo" in client._active_servers
        mock_toolset_class.assert_called_once()
    
    @patch("backend.mcp.client.MCPToolSet")
    def test_stop_server(self, mock_toolset_class, temp_registry_file):
        """Test stopping an MCP server."""
        mock_toolset = MagicMock()
        mock_toolset_class.return_value = mock_toolset
        
        client = MCPClient(registry_path=temp_registry_file, enabled=True)
        client.start_server("test_echo")
        
        result = client.stop_server("test_echo")
        
        assert result is True
        assert "test_echo" not in client._toolsets
        assert "test_echo" not in client._active_servers
        mock_toolset.cleanup.assert_called_once()
    
    @patch("backend.mcp.client.MCPToolSet")
    def test_get_tools(self, mock_toolset_class, mock_toolset, temp_registry_file):
        """Test getting tools from a server."""
        mock_toolset_class.return_value = mock_toolset
        
        client = MCPClient(registry_path=temp_registry_file, enabled=True)
        tools = client.get_tools("test_echo")
        
        assert len(tools) == 1
        assert tools[0].name == "echo"
    
    def test_get_tools_disabled(self, temp_registry_file):
        """Test getting tools when MCP is disabled."""
        client = MCPClient(registry_path=temp_registry_file, enabled=False)
        tools = client.get_tools("test_echo")
        assert len(tools) == 0
    
    @patch("backend.mcp.client.MCPToolSet")
    def test_get_all_tools(self, mock_toolset_class, mock_toolset, temp_registry_file):
        """Test getting all tools from all servers."""
        mock_toolset_class.return_value = mock_toolset
        
        client = MCPClient(registry_path=temp_registry_file, enabled=True)
        tools = client.get_all_tools()
        
        assert len(tools) == 1
        assert tools[0].name == "echo"
    
    @patch("backend.mcp.client.MCPToolSet")
    def test_cleanup(self, mock_toolset_class, temp_registry_file):
        """Test cleaning up all servers."""
        mock_toolset = MagicMock()
        mock_toolset_class.return_value = mock_toolset
        
        client = MCPClient(registry_path=temp_registry_file, enabled=True)
        client.start_server("test_echo")
        
        client.cleanup()
        
        assert len(client._toolsets) == 0
        assert len(client._active_servers) == 0
        mock_toolset.cleanup.assert_called_once()
    
    @patch("backend.mcp.client.MCPToolSet")
    def test_context_manager(self, mock_toolset_class, mock_toolset, temp_registry_file):
        """Test using MCPClient as a context manager."""
        mock_toolset_class.return_value = mock_toolset
        
        with MCPClient(registry_path=temp_registry_file, enabled=True) as client:
            client.start_server("test_echo")
            assert "test_echo" in client._toolsets
        
        # After exiting context, cleanup should have been called
        mock_toolset.cleanup.assert_called_once()
    
    @patch("backend.mcp.client.MCPToolSet")
    def test_server_session(self, mock_toolset_class, mock_toolset, temp_registry_file):
        """Test using server_session context manager."""
        mock_toolset_class.return_value = mock_toolset
        
        client = MCPClient(registry_path=temp_registry_file, enabled=True)
        
        with client.server_session("test_echo") as tools:
            assert len(tools) == 1
            assert tools[0].name == "echo"
            assert "test_echo" in client._toolsets
        
        # After exiting context, server should be stopped
        assert "test_echo" not in client._toolsets
        mock_toolset.cleanup.assert_called_once()


class TestMCPFactory:
    """Tests for MCP factory functions."""
    
    def test_get_mcp_client(self, temp_registry_file):
        """Test get_mcp_client factory function."""
        client = get_mcp_client(registry_path=temp_registry_file, enabled=True)
        assert isinstance(client, MCPClient)
        assert client.registry_path == temp_registry_file
        assert client.enabled is True
    
    @patch("backend.mcp.client.MCPClient")
    def test_get_mcp_tools_for_factory(self, mock_client_class, mock_toolset):
        """Test get_mcp_tools_for_factory function."""
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get_all_tools.return_value = list(mock_toolset)
        mock_client_class.return_value = mock_client
        
        with patch("backend.mcp.client.MCP_ENABLED", True):
            tools = get_mcp_tools_for_factory()
            
            assert len(tools) == 1
            assert tools[0].name == "echo"
            mock_client.get_all_tools.assert_called_once()
    
    @patch("backend.mcp.client.MCPClient")
    def test_get_mcp_tools_for_factory_disabled(self, mock_client_class):
        """Test get_mcp_tools_for_factory when MCP is disabled."""
        with patch("backend.mcp.client.MCP_ENABLED", False):
            tools = get_mcp_tools_for_factory()
            
            assert tools == []
            mock_client_class.assert_not_called()
    
    @patch("backend.mcp.client.MCPClient")
    def test_get_mcp_tools_for_factory_error(self, mock_client_class):
        """Test get_mcp_tools_for_factory when an error occurs."""
        mock_client = MagicMock()
        mock_client.__enter__.side_effect = Exception("Test error")
        mock_client_class.return_value = mock_client
        
        with patch("backend.mcp.client.MCP_ENABLED", True):
            tools = get_mcp_tools_for_factory()
            
            assert tools == []


# Integration Tests
class TestMCPServerIntegration:
    """Integration tests for MCP servers."""
    
    def test_echo_server_schema(self):
        """Test echo server schema."""
        # Get the tool schemas from the server
        tools = echo_server.list_tools()
        
        assert len(tools) >= 1
        assert any(tool["name"] == "echo" for tool in tools)
        
        # Check schema for echo tool
        echo_tool = next(tool for tool in tools if tool["name"] == "echo")
        assert "description" in echo_tool
        assert "inputSchema" in echo_tool
        assert echo_tool["inputSchema"]["properties"]["text"]["type"] == "string"
    
    @pytest.mark.asyncio
    async def test_echo_tool_execution(self):
        """Test executing the echo tool."""
        # Create a context mock
        ctx = MagicMock(spec=Context)
        
        # Get the echo tool function
        echo_tool = None
        for tool in echo_server._tools:
            if tool["name"] == "echo":
                echo_tool = tool["handler"]
                break
        
        assert echo_tool is not None
        
        # Execute the tool
        result = await echo_tool(ctx, "test message")
        
        assert result["success"] is True
        assert result["text"] == "ECHO: test message"
        assert result["original"] == "test message"
    
    @patch("backend.agents.tools.get_all_tools")
    def test_tools_integration(self, mock_get_all_tools, mock_toolset):
        """Test integration with CrewAI tools."""
        # Mock the get_all_tools function to return our mock tools
        mock_get_all_tools.return_value = list(mock_toolset)
        
        # Get all tools
        tools = get_all_tools()
        
        assert len(tools) == 1
        assert tools[0].name == "echo"


@pytest.mark.asyncio
class TestMCPGeminiIntegration:
    """Tests for MCP integration with Gemini."""
    
    @patch("google.genai.Client")
    @patch("backend.mcp.client.MCPClient")
    async def test_gemini_mcp_integration(self, mock_client_class, mock_gemini_class, mock_toolset, mock_gemini_client):
        """Test integration between Gemini and MCP."""
        # Mock the MCP client
        mock_mcp_client = MagicMock()
        mock_mcp_client.__enter__.return_value = mock_mcp_client
        mock_mcp_client.get_tools.return_value = list(mock_toolset)
        mock_client_class.return_value = mock_mcp_client
        
        # Mock the Gemini client
        mock_gemini_class.return_value = mock_gemini_client
        
        # Set up environment
        os.environ["ENABLE_MCP"] = "1"
        
        # Import the demo script
        from scripts.mcp_demo import run_echo_demo
        
        # Run the echo demo
        await run_echo_demo(mock_mcp_client)
        
        # Check that Gemini was called with the correct parameters
        mock_gemini_client.aio.models.generate_content.assert_called_once()
        
        # Get the call arguments
        args, kwargs = mock_gemini_client.aio.models.generate_content.call_args
        
        # Check that the model and tools were passed correctly
        assert kwargs["model"] == "gemini-2.5-flash"
        assert "config" in kwargs
        assert kwargs["config"].temperature == 0
        assert kwargs["config"].tools == list(mock_toolset)
    
    @pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    @pytest.mark.skipif(not os.environ.get("ENABLE_MCP"), reason="ENABLE_MCP not set")
    async def test_live_gemini_mcp_integration(self):
        """
        Test live integration between Gemini and MCP.
        
        This test requires:
        - GEMINI_API_KEY environment variable to be set
        - ENABLE_MCP=1 environment variable to be set
        
        It will be skipped if these are not available.
        """
        from google import genai
        
        # Initialize MCP client with echo server
        with MCPClient() as mcp_client:
            # Get echo tools
            tools = mcp_client.get_tools("echo")
            
            if not tools:
                pytest.skip("Echo server tools not available")
            
            # Initialize Gemini client
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            
            # Create a simple prompt
            prompt = "Please use the echo tool to echo back the text 'test message'"
            
            # Generate content with MCP tools
            from google.genai import types
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=tools,
                )
            )
            
            # Check that the response contains the echoed message
            assert response is not None
            assert "test message" in response.text.lower()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
