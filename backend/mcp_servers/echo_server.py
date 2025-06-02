"""
Echo MCP Server - Simple test server for MCP proof-of-concept

This is a lightweight MCP server that implements a simple echo tool for testing
the Model Context Protocol integration with Gemini and CrewAI. It follows the
pattern from the MCP technical guide and uses the mcpengine framework.

Usage:
    python backend/mcp_servers/echo_server.py

The server will start and listen for JSON-RPC requests on stdin/stdout.
"""

import logging
from typing import Dict, Any, Optional
from mcpengine import Server, Tool, Context

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("echo-server")

# Initialize the MCP server
server = Server(
    name="echo-server",
    description="Simple echo server for MCP testing"
)

@server.tool(
    name="echo",
    description="Return whatever string the user sends back",
    input_schema={
        "type": "object", 
        "properties": {
            "text": {"type": "string", "description": "Text to echo back"}
        }, 
        "required": ["text"]
    }
)
async def echo(ctx: Context, text: str) -> Dict[str, Any]:
    """
    Echo back the provided text with 'ECHO: ' prefix.
    
    Args:
        ctx: MCP context
        text: Text to echo back
        
    Returns:
        Dictionary with the echoed text and timestamp
    """
    logger.info(f"Received echo request: {text}")
    try:
        result = {
            "success": True,
            "text": f"ECHO: {text}",
            "original": text,
            "length": len(text)
        }
        return result
    except Exception as e:
        logger.error(f"Error in echo tool: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@server.tool(
    name="echo_with_metadata",
    description="Echo text back with additional metadata",
    input_schema={
        "type": "object", 
        "properties": {
            "text": {"type": "string", "description": "Text to echo back"},
            "include_stats": {"type": "boolean", "description": "Whether to include text statistics"}
        }, 
        "required": ["text"]
    }
)
async def echo_with_metadata(ctx: Context, text: str, include_stats: Optional[bool] = False) -> Dict[str, Any]:
    """
    Echo text back with additional metadata.
    
    Args:
        ctx: MCP context
        text: Text to echo back
        include_stats: Whether to include text statistics
        
    Returns:
        Dictionary with echoed text and metadata
    """
    logger.info(f"Received echo with metadata request: {text}, stats={include_stats}")
    
    result = {
        "success": True,
        "text": f"ECHO: {text}",
        "original": text,
        "length": len(text)
    }
    
    if include_stats:
        # Add some basic text statistics
        result["stats"] = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "uppercase_count": sum(1 for c in text if c.isupper()),
            "lowercase_count": sum(1 for c in text if c.islower()),
            "digit_count": sum(1 for c in text if c.isdigit()),
            "space_count": sum(1 for c in text if c.isspace())
        }
    
    return result

if __name__ == "__main__":
    """Run the MCP server."""
    logger.info("Starting Echo MCP Server...")
    server.run()
