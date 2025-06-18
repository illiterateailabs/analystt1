"""
Sim API Balances Tool

This tool integrates with Sim APIs to fetch token balances for any wallet address
across 60+ EVM chains. It provides comprehensive details about native and ERC20 tokens,
including token metadata and USD valuations.

API Reference: https://docs.sim.dune.com/evm/balances
"""

import time
import logging
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.core.events import emit_event, GraphAddEvent
from backend.agents.tools.base_tool import BaseTool
from backend.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Schema definitions for validation and type safety
class TokenMetadata(BaseModel):
    """Token metadata including symbol, name, decimals, and optional fields."""
    symbol: str
    name: Optional[str] = None
    decimals: int
    logo: Optional[str] = None
    url: Optional[str] = None

class TokenBalance(BaseModel):
    """Token balance with metadata and value information."""
    address: str
    amount: str
    chain: str
    chain_id: int
    decimals: int
    name: Optional[str] = None
    symbol: str
    price_usd: Optional[float] = None
    value_usd: Optional[float] = None
    token_metadata: Optional[TokenMetadata] = None
    low_liquidity: Optional[bool] = None
    pool_size: Optional[float] = None
    
    @validator('amount')
    def validate_amount(cls, v):
        """Ensure amount is a valid numeric string."""
        try:
            # Just check if it can be parsed as a number
            float(v)
            return v
        except ValueError:
            raise ValueError("amount must be a valid numeric string")

class BalancesResponse(BaseModel):
    """Response structure from the Sim Balances API."""
    balances: List[TokenBalance]
    wallet_address: str
    next_offset: Optional[str] = None
    request_time: Optional[str] = None
    response_time: Optional[str] = None

class SimBalancesTool(BaseTool):
    """
    Tool for fetching token balances from Sim APIs.
    
    This tool retrieves native and ERC20 token balances for any wallet address
    across 60+ EVM chains. It includes token metadata, USD valuations, and
    liquidity information.
    """
    
    name = "sim_balances_tool"
    description = """
    Fetches token balances for any wallet address across 60+ EVM chains.
    Returns native and ERC20 token balances with USD values, token metadata,
    and liquidity information.
    """
    
    def __init__(self):
        """Initialize the Sim Balances tool with API configuration."""
        super().__init__()
        self.api_url = settings.SIM_API_URL
        self.api_key = settings.SIM_API_KEY
        self.headers = {
            "X-Sim-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.HTTPError)),
        reraise=True
    )
    def _make_request(self, url: str) -> Dict[str, Any]:
        """
        Make a request to the Sim API with retry logic.
        
        Args:
            url: The full URL for the API request
            
        Returns:
            The JSON response as a dictionary
            
        Raises:
            requests.exceptions.HTTPError: If the request fails after retries
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit exceeded
                logger.warning("Rate limit exceeded. Implementing backoff strategy.")
                # The @retry decorator will handle the retry with exponential backoff
                raise
            elif e.response.status_code >= 500:  # Server errors
                logger.error(f"Server error from Sim API: {e}")
                raise
            else:  # Client errors
                logger.error(f"Client error when calling Sim API: {e}")
                # Try to get error details from response
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {error_details}")
                except:
                    pass
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception when calling Sim API: {e}")
            raise
    
    def run(
        self, 
        wallet: str, 
        limit: int = 100, 
        offset: Optional[str] = None,
        chains: str = "all",
        metadata: str = "url,logo",
        emit_graph_events: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch token balances for a wallet address.
        
        Args:
            wallet: The wallet address to query
            limit: Maximum number of balances to return (default: 100)
            offset: Pagination offset token from previous response
            chains: Comma-separated list of chain IDs or "all" for all chains
            metadata: Comma-separated list of metadata to include (url, logo)
            emit_graph_events: Whether to emit graph events for Neo4j integration
            
        Returns:
            Dictionary containing token balances and pagination info
            
        Raises:
            ValueError: If wallet address is invalid
            Exception: If API request fails after retries
        """
        if not wallet or not isinstance(wallet, str):
            raise ValueError("Wallet address must be a non-empty string")
        
        # Build query parameters
        params = []
        if limit:
            params.append(f"limit={limit}")
        if offset:
            params.append(f"offset={offset}")
        if chains:
            params.append(f"chain_ids={chains}")
        if metadata:
            params.append(f"metadata={metadata}")
        
        query_string = "&".join(params)
        url = f"{self.api_url}/v1/evm/balances/{wallet}"
        if query_string:
            url = f"{url}?{query_string}"
        
        try:
            # Make the API request
            response_data = self._make_request(url)
            
            # Validate response with Pydantic
            validated_response = BalancesResponse(**response_data)
            
            # Emit graph events if requested
            if emit_graph_events and validated_response.balances:
                self._emit_graph_events(wallet, validated_response.balances)
            
            # Return the validated response as a dictionary
            return validated_response.dict()
            
        except Exception as e:
            logger.error(f"Error fetching balances for wallet {wallet}: {str(e)}")
            raise
    
    def _emit_graph_events(self, wallet: str, balances: List[TokenBalance]) -> None:
        """
        Emit graph events for Neo4j integration.
        
        Args:
            wallet: The wallet address
            balances: List of token balances
        """
        try:
            # Prepare data for Neo4j
            graph_data = {
                "wallet": wallet,
                "balances": [balance.dict() for balance in balances],
                "timestamp": int(time.time())
            }
            
            # Emit event for graph processing
            emit_event(
                GraphAddEvent(
                    type="wallet_balances",
                    data=graph_data
                )
            )
            logger.debug(f"Emitted graph events for wallet {wallet} with {len(balances)} balances")
        except Exception as e:
            logger.error(f"Failed to emit graph events: {str(e)}")
            # Don't re-raise, as this is a non-critical operation
