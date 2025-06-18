"""
SimSVMBalancesTool - Fetch Solana (SVM) token balances from Sim APIs.

This tool retrieves native SOL and SPL token balances for a given wallet address
on Solana-based chains (Solana, Eclipse) using the Sim APIs. It supports pagination,
chain filtering, and includes comprehensive token metadata and USD pricing.
It's designed for financial crime analysis, portfolio tracking, and identifying
asset holdings on SVM chains.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from crewai_tools import BaseTool
from pydantic import BaseModel, Field, validator

from backend.integrations.sim_client import SimClient
from backend.core.metrics import record_tool_usage, record_tool_error

logger = logging.getLogger(__name__)


class SimSVMBalancesInput(BaseModel):
    """Input schema for SimSVMBalancesTool."""
    
    address: str = Field(
        ...,
        description="The wallet address to fetch Solana (SVM) token balances for."
    )
    limit: int = Field(
        50,
        description="Maximum number of balances to return (default: 50, max: 100)."
    )
    offset: Optional[str] = Field(
        None,
        description="Pagination cursor for fetching the next page of results."
    )
    chains: str = Field(
        "all",
        description="Comma-separated list of chains to filter by (e.g., 'solana,eclipse'), or 'all' for all supported SVM chains."
    )

    @validator('chains')
    def validate_chains(cls, v):
        if v != "all":
            valid_chains = {'solana', 'eclipse'}
            requested_chains = {chain.strip().lower() for chain in v.split(',')}
            invalid_chains = requested_chains - valid_chains
            if invalid_chains:
                raise ValueError(f"Invalid chains: {', '.join(invalid_chains)}. Valid options are: solana, eclipse, or 'all'")
        return v


class SimSVMBalancesTool(BaseTool):
    """
    Tool for retrieving native SOL and SPL token balances for a Solana (SVM) wallet address.
    
    This tool fetches detailed balance information, including token metadata, USD pricing,
    and liquidity status. It's essential for comprehensive asset tracing and financial
    analysis on Solana-based ecosystems.
    """
    
    name: str = "sim_svm_balances_tool"
    description: str = (
        "Fetch Solana (SVM) token balances (native SOL and SPL tokens) for a wallet address "
        "across specified SVM chains (e.g., 'solana', 'eclipse', or 'all'). "
        "Includes token metadata, USD pricing, and total portfolio value."
    )
    
    sim_client: SimClient
    
    def __init__(self, sim_client: Optional[SimClient] = None):
        """
        Initialize the SimSVMBalancesTool.
        
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
        chains: str = "all",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch Solana (SVM) token balances for the specified wallet address.
        
        Args:
            address: The wallet address to fetch balances for.
            limit: Maximum number of balances to return (default: 50, max: 100).
            offset: Pagination cursor for fetching next page of results.
            chains: Comma-separated list of chains (e.g., 'solana,eclipse'), or 'all'.
            
        Returns:
            Dictionary containing balances data, pagination info, and summary.
            
        Raises:
            Exception: If there's an error fetching the balances.
        """
        try:
            record_tool_usage(self.name)
            
            # Prepare query parameters
            params = {"limit": min(limit, 100)}  # Cap at 100 to prevent abuse
            
            if offset:
                params["offset"] = offset
                
            if chains and chains != "all":
                params["chains"] = chains
            
            # Make API request
            logger.info(f"Fetching SVM balances for address {address} on chains {chains} with params {params}")
            response = await self.sim_client.get(f"/beta/svm/balances/{address}", **params)
            
            # Process and structure the response
            balances = response.get("balances", [])
            next_offset = response.get("next_offset")
            balances_count = response.get("balances_count", len(balances))
            
            total_portfolio_value_usd = self._calculate_total_portfolio_value(balances)
            
            # Create summary information
            chain_summary = {}
            token_types_summary = {"native": 0, "spl": 0}
            
            for balance in balances:
                # Count by chain
                chain = balance.get("chain", "unknown")
                chain_summary[chain] = chain_summary.get(chain, 0) + 1
                
                # Count by token type
                if balance.get("address") == "native" or balance.get("is_native", False):
                    token_types_summary["native"] += 1
                else:
                    token_types_summary["spl"] += 1

            result = {
                "balances": balances,
                "count": len(balances),
                "total_balances_count": balances_count,
                "next_offset": next_offset,
                "has_more": next_offset is not None,
                "wallet_address": address,
                "total_portfolio_value_usd": total_portfolio_value_usd,
                "request_time": response.get("processing_time_ms"),
                "summary": {
                    "chains": chain_summary,
                    "token_types": token_types_summary,
                    "total_value_usd": total_portfolio_value_usd
                }
            }
            
            # Handle empty results gracefully
            if not balances:
                logger.info(f"No SVM balances found for address {address} on chains {chains}")
                result["message"] = "No SVM balances found for this wallet on the specified chains."
            
            return result
            
        except Exception as e:
            error_msg = f"Error fetching SVM balances for {address}: {str(e)}"
            logger.error(error_msg)
            record_tool_error(self.name, str(e))
            
            # Return structured error response
            return {
                "error": error_msg,
                "balances": [],
                "count": 0,
                "wallet_address": address,
                "total_portfolio_value_usd": 0.0
            }

    def _calculate_total_portfolio_value(self, balances: List[Dict[str, Any]]) -> float:
        """
        Calculate the total portfolio value in USD from a list of balances.
        
        Args:
            balances: List of balance objects from the Sim API response.
            
        Returns:
            Total portfolio value in USD, rounded to 2 decimal places.
        """
        total_value = 0.0
        for balance in balances:
            value_usd = balance.get("value_usd")
            if value_usd is not None:
                try:
                    total_value += float(value_usd)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert value_usd to float: {value_usd}")
        return round(total_value, 2)
