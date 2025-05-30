"""
Crypto tools module for blockchain data analysis.

This module provides tools for interacting with various blockchain data sources,
including Dune Analytics, DefiLlama, and Etherscan. These tools can be used by
agents to retrieve and analyze blockchain data for crypto and DeFi analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Type

from backend.integrations.neo4j_client import Neo4jClient

# Import all crypto tools
from .dune_analytics_tool import DuneAnalyticsTool
from .defillama_tool import DefiLlamaTool
from .etherscan_tool import EtherscanTool

logger = logging.getLogger(__name__)

# Tool registry with metadata
CRYPTO_TOOLS = {
    "dune_analytics_tool": {
        "class": DuneAnalyticsTool,
        "description": "Execute SQL queries against blockchain data using Dune Analytics",
        "requires_api_key": True,
        "api_key_setting": "dune_api_key",
        "data_sources": ["ethereum", "polygon", "solana", "bitcoin", "arbitrum", "optimism"],
        "capabilities": ["sql_queries", "custom_analytics", "multi_chain_analysis"]
    },
    "defillama_tool": {
        "class": DefiLlamaTool,
        "description": "Analyze DeFi protocols, TVL, yields, and stablecoins using DefiLlama data",
        "requires_api_key": False,
        "data_sources": ["defi_protocols", "yields", "stablecoins"],
        "capabilities": ["tvl_analysis", "yield_tracking", "protocol_comparison"]
    },
    "etherscan_tool": {
        "class": EtherscanTool,
        "description": "Analyze blockchain data using Etherscan and similar block explorers",
        "requires_api_key": True,
        "api_key_setting": "etherscan_api_key",
        "data_sources": ["ethereum", "bsc", "polygon", "arbitrum", "optimism", "fantom", "avalanche"],
        "capabilities": ["address_analysis", "token_tracking", "contract_analysis"]
    }
}

# Export all tool classes
__all__ = [
    "DuneAnalyticsTool",
    "DefiLlamaTool", 
    "EtherscanTool",
    "create_crypto_tools",
    "get_tool_metadata",
    "CRYPTO_TOOLS"
]

def create_crypto_tools(
    neo4j_client: Optional[Neo4jClient] = None,
    api_keys: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Create instances of all crypto tools.
    
    Args:
        neo4j_client: Neo4j client for storing results
        api_keys: Dictionary of API keys for tools that require them
    
    Returns:
        Dictionary of tool instances by name
    """
    api_keys = api_keys or {}
    tools = {}
    
    for tool_name, tool_info in CRYPTO_TOOLS.items():
        try:
            tool_class = tool_info["class"]
            
            # Handle tools that require API keys
            if tool_info.get("requires_api_key", False):
                api_key_setting = tool_info.get("api_key_setting")
                api_key = api_keys.get(api_key_setting)
                
                if api_key:
                    tools[tool_name] = tool_class(api_key=api_key, neo4j_client=neo4j_client)
                else:
                    logger.warning(f"No API key provided for {tool_name}, tool may not function properly")
                    tools[tool_name] = tool_class(neo4j_client=neo4j_client)
            else:
                # Tools that don't require API keys
                tools[tool_name] = tool_class(neo4j_client=neo4j_client)
                
            logger.info(f"Created crypto tool: {tool_name}")
            
        except Exception as e:
            logger.error(f"Error creating crypto tool {tool_name}: {e}")
    
    return tools

def get_tool_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all crypto tools.
    
    Returns:
        Dictionary of tool metadata by name
    """
    return CRYPTO_TOOLS
