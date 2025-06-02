#!/usr/bin/env python
"""
MCP Demo Script for analystt1

This script demonstrates the Model Context Protocol (MCP) integration with
Gemini 2.5 Flash. It connects to MCP servers, shows tool discovery, and
executes tools through the Gemini API.

Usage:
    python scripts/mcp_demo.py

Requirements:
    - GEMINI_API_KEY environment variable must be set
    - Neo4j database connection for graph server (if using graph tools)

Example:
    ENABLE_MCP=1 GEMINI_API_KEY=your_key python scripts/mcp_demo.py
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types

from backend.mcp import MCPClient
from backend.core.logging import get_logger

# Configure logging
logger = get_logger("mcp_demo")
logger.setLevel(logging.INFO)

# Check for required environment variables
if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Enable MCP
os.environ["ENABLE_MCP"] = "1"

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def list_available_tools(mcp_client: MCPClient) -> Dict[str, List[str]]:
    """
    List all available tools from MCP servers.
    
    Args:
        mcp_client: MCP client instance
        
    Returns:
        Dictionary mapping server names to lists of tool names
    """
    tools_by_server = {}
    
    # Get tools from each server
    for server_name in ["echo", "graph"]:
        try:
            tools = mcp_client.get_tools(server_name)
            tools_by_server[server_name] = [tool.name for tool in tools]
            logger.info(f"Server '{server_name}' provides tools: {', '.join(tools_by_server[server_name])}")
        except Exception as e:
            logger.error(f"Error getting tools from server '{server_name}': {e}")
            tools_by_server[server_name] = []
    
    return tools_by_server


async def run_echo_demo(mcp_client: MCPClient) -> None:
    """
    Run a demo with the echo server.
    
    Args:
        mcp_client: MCP client instance
    """
    logger.info("=== Running Echo Server Demo ===")
    
    try:
        # Get echo tools
        with mcp_client.server_session("echo") as session:
            if not session:
                logger.error("Failed to start echo server session")
                return
            
            # Initialize the Gemini model
            model = "gemini-2.5-flash"
            
            # Prompt to use the echo tool
            prompt = "Please use the echo tool to echo back the text 'Hello from MCP!'"
            
            logger.info(f"Sending prompt to {model}: {prompt}")
            
            # Generate content with MCP tools
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=session,  # Pass MCP session as tools
                )
            )
            
            # Print the response
            logger.info(f"Response from Gemini:\n{response.text}")
    
    except Exception as e:
        logger.error(f"Error in echo demo: {e}")


async def run_graph_demo(mcp_client: MCPClient) -> None:
    """
    Run a demo with the graph server.
    
    Args:
        mcp_client: MCP client instance
    """
    logger.info("=== Running Graph Server Demo ===")
    
    try:
        # Get graph tools
        with mcp_client.server_session("graph") as session:
            if not session:
                logger.error("Failed to start graph server session")
                return
            
            # Initialize the Gemini model
            model = "gemini-2.5-flash"
            
            # Prompt to use the graph query tool
            prompt = (
                "Please use the cypher_query tool to run this query against Neo4j: "
                "MATCH (n) RETURN count(n) as nodeCount LIMIT 1"
            )
            
            logger.info(f"Sending prompt to {model}: {prompt}")
            
            # Generate content with MCP tools
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=session,  # Pass MCP session as tools
                )
            )
            
            # Print the response
            logger.info(f"Response from Gemini:\n{response.text}")
    
    except Exception as e:
        logger.error(f"Error in graph demo: {e}")


async def run_multi_tool_demo(mcp_client: MCPClient) -> None:
    """
    Run a demo with multiple tools from different servers.
    
    Args:
        mcp_client: MCP client instance
    """
    logger.info("=== Running Multi-Tool Demo ===")
    
    try:
        # Get all tools
        all_tools = mcp_client.get_all_tools()
        
        if not all_tools:
            logger.error("No tools available")
            return
        
        logger.info(f"Using {len(all_tools)} tools from all servers")
        
        # Initialize the Gemini model
        model = "gemini-2.5-flash"
        
        # Prompt to use multiple tools
        prompt = (
            "I need your help with two tasks:\n"
            "1. Use the echo tool to echo back 'Multi-tool test'\n"
            "2. Then use the cypher_query tool to count the number of relationships "
            "in the database with: MATCH ()-[r]->() RETURN count(r) as relCount LIMIT 1"
        )
        
        logger.info(f"Sending prompt to {model}: {prompt}")
        
        # Generate content with MCP tools
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                tools=all_tools,  # Pass all tools
            )
        )
        
        # Print the response
        logger.info(f"Response from Gemini:\n{response.text}")
    
    except Exception as e:
        logger.error(f"Error in multi-tool demo: {e}")


async def run() -> None:
    """Main function to run the MCP demo."""
    logger.info(f"Starting MCP Demo at {datetime.now().isoformat()}")
    logger.info(f"Using Gemini API with key: {os.getenv('GEMINI_API_KEY')[:5]}...")
    
    try:
        # Initialize MCP client
        with MCPClient() as mcp_client:
            # List available tools
            tools_by_server = await list_available_tools(mcp_client)
            
            # Run echo demo
            await run_echo_demo(mcp_client)
            
            # Run graph demo
            await run_graph_demo(mcp_client)
            
            # Run multi-tool demo
            await run_multi_tool_demo(mcp_client)
            
            logger.info("MCP Demo completed successfully")
    
    except Exception as e:
        logger.error(f"Error in MCP Demo: {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(run())
