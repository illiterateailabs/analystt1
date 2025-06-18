"""
SimCollectiblesTool - Fetch NFT collectibles from Sim APIs.

This tool retrieves ERC721 and ERC1155 NFT collectibles owned by a wallet address
across multiple EVM chains using the Sim APIs. It supports pagination, chain filtering,
and provides structured data for financial crime analysis and NFT ownership tracking.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

from backend.integrations.sim_client import SimClient
from backend.core.metrics import record_tool_usage, record_tool_error

logger = logging.getLogger(__name__)


class SimCollectiblesInput(BaseModel):
    """Input schema for SimCollectiblesTool."""
    
    address: str = Field(
        ...,
        description="The wallet address to fetch NFT collectibles for"
    )
    limit: int = Field(
        50,
        description="Maximum number of collectibles to return (default: 50, max: 100)"
    )
    offset: Optional[str] = Field(
        None,
        description="Pagination cursor for fetching next page of results"
    )
    chain_ids: Optional[str] = Field(
        None,
        description="Comma-separated list of chain IDs to filter by, or 'all' for all supported chains"
    )


class SimCollectiblesTool(BaseTool):
    """
    Tool for retrieving NFT collectibles (ERC721 and ERC1155) owned by a wallet address.
    
    This tool fetches collectibles across multiple EVM chains, providing detailed
    information about each NFT including token ID, contract address, balance,
    and metadata when available. It's useful for financial crime analysis,
    tracking NFT movements, and identifying patterns in digital asset ownership.
    """
    
    name: str = "sim_collectibles_tool"
    description: str = "Fetch NFT collectibles (ERC721 and ERC1155) owned by a wallet address across multiple EVM chains"
    
    sim_client: SimClient
    
    def __init__(self, sim_client: Optional[SimClient] = None):
        """
        Initialize the SimCollectiblesTool.
        
        Args:
            sim_client: Optional SimClient instance. If not provided, a new one will be created.
        """
        super().__init__()
        self.sim_client = sim_client or SimClient()
    
    async def _execute(
        self,
        address: str,
        limit: int = 50,
        offset: Optional[str] = None,
        chain_ids: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch NFT collectibles for the specified wallet address.
        
        Args:
            address: The wallet address to fetch collectibles for
            limit: Maximum number of collectibles to return (default: 50, max: 100)
            offset: Pagination cursor for fetching next page of results
            chain_ids: Comma-separated list of chain IDs to filter by, or 'all' for all supported chains
            
        Returns:
            Dictionary containing collectibles data and pagination info
            
        Raises:
            Exception: If there's an error fetching the collectibles
        """
        try:
            # Record tool usage for metrics
            record_tool_usage(self.name)
            
            # Prepare query parameters
            params = {"limit": min(limit, 100)}  # Cap at 100 to prevent abuse
            
            if offset:
                params["offset"] = offset
                
            if chain_ids:
                params["chain_ids"] = chain_ids
            
            # Make API request
            logger.info(f"Fetching collectibles for address {address} with params {params}")
            response = await self.sim_client.get(f"/v1/evm/collectibles/{address}", **params)
            
            # Process and structure the response
            collectibles = response.get("entries", [])
            next_offset = response.get("next_offset")
            
            # Enhance the response with additional context
            result = {
                "collectibles": collectibles,
                "count": len(collectibles),
                "next_offset": next_offset,
                "has_more": next_offset is not None,
                "wallet_address": address,
                "request_time": response.get("request_time"),
                "response_time": response.get("response_time")
            }
            
            # Add summary information
            collections = {}
            chains = {}
            standards = {"ERC721": 0, "ERC1155": 0, "UNKNOWN": 0}
            
            for item in collectibles:
                # Count by collection
                contract = item.get("contract_address", "unknown")
                collections[contract] = collections.get(contract, 0) + 1
                
                # Count by chain
                chain = item.get("chain", "unknown")
                chains[chain] = chains.get(chain, 0) + 1
                
                # Count by standard
                standard = item.get("token_standard", "UNKNOWN")
                standards[standard] = standards.get(standard, 0) + 1
            
            # Add summary to result
            result["summary"] = {
                "unique_collections": len(collections),
                "chains": chains,
                "standards": standards
            }
            
            # Handle empty results gracefully
            if not collectibles:
                logger.info(f"No collectibles found for address {address}")
                result["message"] = "No collectibles found for this wallet"
            
            return result
            
        except Exception as e:
            error_msg = f"Error fetching collectibles for {address}: {str(e)}"
            logger.error(error_msg)
            record_tool_error(self.name, str(e))
            
            # Return structured error response
            return {
                "error": error_msg,
                "collectibles": [],
                "count": 0,
                "wallet_address": address
            }
