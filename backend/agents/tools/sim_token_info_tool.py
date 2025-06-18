"""
SimTokenInfoTool - Fetch detailed metadata and pricing information for tokens.

This tool retrieves comprehensive information about tokens (ERC20 and native)
including their metadata, real-time pricing, and liquidity across multiple EVM chains
using the Sim APIs. It supports both single token lookups and batch queries,
and provides risk assessment based on liquidity.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from crewai_tools import BaseTool
from pydantic import BaseModel, Field, validator

from backend.integrations.sim_client import SimClient
from backend.core.metrics import record_tool_usage, record_tool_error

logger = logging.getLogger(__name__)


class SimTokenInfoInput(BaseModel):
    """Input schema for SimTokenInfoTool."""
    
    token_addresses: Union[str, List[str]] = Field(
        ...,
        description="Single token address or a comma-separated string/list of token addresses to fetch info for."
                    "For native tokens, use '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'."
    )
    chain_ids: str = Field(
        ...,
        description="Comma-separated list of chain IDs to filter by (e.g., '1,137' for Ethereum and Polygon)."
                    "This parameter is mandatory for the Sim API token-info endpoint."
    )
    limit: int = Field(
        50,
        description="Maximum number of results to return per token address (default: 50, max: 100)."
                    "Useful for pagination if a token exists on multiple chains or has many variants."
    )
    offset: Optional[str] = Field(
        None,
        description="Pagination cursor for fetching the next page of results."
    )

    @validator('token_addresses', pre=True)
    def _split_token_addresses(cls, v):
        if isinstance(v, str):
            return [addr.strip() for addr in v.split(',')]
        return v


class SimTokenInfoTool(BaseTool):
    """
    Tool for retrieving detailed information about tokens (ERC20 and native).
    
    This tool fetches token metadata (symbol, name, decimals, supply, logo),
    real-time pricing, and liquidity information. It can handle both single
    token lookups and batch queries, and provides basic risk assessment
    based on liquidity.
    """
    
    name: str = "sim_token_info_tool"
    description: str = (
        "Fetch detailed metadata and pricing information for tokens (ERC20 and native) "
        "across specified EVM chains. Supports single and batch queries. "
        "Mandatory 'chain_ids' parameter (e.g., '1,137'). "
        "For native tokens, use '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee' as the address."
    )
    
    sim_client: SimClient
    
    def __init__(self, sim_client: Optional[SimClient] = None):
        """
        Initialize the SimTokenInfoTool.
        
        Args:
            sim_client: Optional SimClient instance. If not provided, a new one will be created.
        """
        super().__init__()
        self.sim_client = sim_client or SimClient()
    
    async def _execute(
        self,
        token_addresses: Union[str, List[str]],
        chain_ids: str,
        limit: int = 50,
        offset: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch detailed token information for the specified token address(es).
        
        Args:
            token_addresses: Single token address or a comma-separated string/list of token addresses.
                             For native tokens, use '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'.
            chain_ids: Comma-separated list of chain IDs (e.g., '1,137'). Mandatory.
            limit: Maximum number of results to return per token address.
            offset: Pagination cursor for fetching next page of results.
            
        Returns:
            Dictionary containing token information, including metadata, pricing,
            liquidity, and risk assessment.
            
        Raises:
            Exception: If there's an error fetching the token information.
        """
        if isinstance(token_addresses, str):
            token_addresses = [addr.strip() for addr in token_addresses.split(',')]

        all_token_info = []
        errors = []

        for address in token_addresses:
            try:
                record_tool_usage(self.name)
                
                params = {
                    "chain_ids": chain_ids,
                    "limit": min(limit, 100)  # Cap at 100 to prevent abuse
                }
                if offset:
                    params["offset"] = offset
                
                logger.info(f"Fetching token info for {address} on chains {chain_ids} with params {params}")
                response = await self.sim_client.get(f"/v1/evm/token-info/{address}", **params)
                
                # Process and structure the response
                token_data = response.get("entries", [])
                next_offset = response.get("next_offset")

                for token in token_data:
                    # Add risk assessment
                    token["risk_assessment"] = self._assess_token_risk(token)
                    all_token_info.append(token)
                
                if not token_data:
                    logger.info(f"No token info found for address {address} on chains {chain_ids}")
                    errors.append(f"No token info found for {address} on chains {chain_ids}")

            except Exception as e:
                error_msg = f"Error fetching token info for {address}: {str(e)}"
                logger.error(error_msg)
                record_tool_error(self.name, str(e))
                errors.append(error_msg)
        
        result = {
            "token_info": all_token_info,
            "count": len(all_token_info),
            "next_offset": next_offset if len(token_addresses) == 1 else None, # Only provide if single query
            "has_more": next_offset is not None if len(token_addresses) == 1 else False,
            "query_addresses": token_addresses,
            "chain_ids": chain_ids,
            "errors": errors if errors else None
        }

        if not all_token_info and errors:
            result["message"] = "No token info found due to errors."
        elif not all_token_info:
            result["message"] = "No token info found for the given criteria."
            
        return result

    def _assess_token_risk(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the risk of a token based on liquidity and other factors.
        """
        risk_factors = []
        risk_score = 0

        # Liquidity assessment
        low_liquidity = token_data.get("low_liquidity", False)
        pool_size_usd = token_data.get("pool_size_usd")

        if low_liquidity:
            risk_factors.append("Low liquidity detected.")
            risk_score += 3

        if pool_size_usd is not None and pool_size_usd < 100000:  # Example threshold
            risk_factors.append(f"Low pool size (${pool_size_usd:,.2f}).")
            risk_score += 2
        
        # Check for missing metadata (could indicate scam/unverified token)
        if not token_data.get("name") or not token_data.get("symbol"):
            risk_factors.append("Missing essential metadata (name/symbol).")
            risk_score += 1

        # Check for very low price (potential dead coin or scam)
        price_usd = token_data.get("price_usd")
        if price_usd is not None and price_usd < 0.000001: # Example threshold
            risk_factors.append("Extremely low price, potential dead coin or scam.")
            risk_score += 1

        return {
            "score": min(risk_score, 5), # Cap score at 5
            "factors": risk_factors if risk_factors else ["No significant risks detected."],
            "is_risky": risk_score > 0
        }
