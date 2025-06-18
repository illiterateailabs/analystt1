"""
Sim API Activity Tool

This tool integrates with Sim APIs to fetch chronologically ordered transactions
for any wallet address across 60+ EVM chains. It provides comprehensive details about
native transfers, ERC20 movements, NFT transfers, and decoded contract interactions.

API Reference: https://docs.sim.dune.com/evm/activity
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
    """Token metadata including symbol, name, and decimals."""
    symbol: Optional[str] = None
    name: Optional[str] = None
    decimals: Optional[int] = None
    logo: Optional[str] = None

class FunctionParameter(BaseModel):
    """Function parameter details for decoded contract interactions."""
    name: str
    type: str
    value: Any

class FunctionInfo(BaseModel):
    """Decoded function call information."""
    name: str
    signature: Optional[str] = None
    parameters: Optional[List[FunctionParameter]] = None

class ActivityItem(BaseModel):
    """Individual activity item with transaction details."""
    id: Optional[str] = None
    type: str  # send, receive, mint, burn, swap, approve, call
    chain: str
    chain_id: int
    block_number: int
    block_time: str
    transaction_hash: str
    from_address: Optional[str] = None
    to_address: Optional[str] = None
    asset_type: Optional[str] = None  # native, erc20, erc721, etc.
    amount: Optional[str] = None
    value: Optional[str] = None
    value_usd: Optional[float] = None
    token_address: Optional[str] = None
    token_id: Optional[str] = None
    token_metadata: Optional[TokenMetadata] = None
    function: Optional[FunctionInfo] = None
    
    @validator('type')
    def validate_type(cls, v):
        """Validate activity type."""
        valid_types = ['send', 'receive', 'mint', 'burn', 'swap', 'approve', 'call']
        if v not in valid_types:
            raise ValueError(f"Activity type must be one of: {', '.join(valid_types)}")
        return v

class ActivityResponse(BaseModel):
    """Response structure from the Sim Activity API."""
    activity: List[ActivityItem]
    wallet_address: str
    next_offset: Optional[str] = None
    request_time: Optional[str] = None
    response_time: Optional[str] = None

class SimActivityTool(BaseTool):
    """
    Tool for fetching transaction activity from Sim APIs.
    
    This tool retrieves chronologically ordered transactions for any wallet address
    across 60+ EVM chains. It includes native transfers, ERC20 movements, NFT transfers,
    and decoded contract interactions.
    """
    
    name = "sim_activity_tool"
    description = """
    Fetches transaction activity for any wallet address across 60+ EVM chains.
    Returns chronologically ordered transactions including native transfers, 
    ERC20 movements, NFT transfers, and decoded contract interactions.
    """
    
    def __init__(self):
        """Initialize the Sim Activity tool with API configuration."""
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
        limit: int = 25, 
        offset: Optional[str] = None,
        emit_graph_events: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch transaction activity for a wallet address.
        
        Args:
            wallet: The wallet address to query
            limit: Maximum number of activities to return (default: 25)
            offset: Pagination offset token from previous response
            emit_graph_events: Whether to emit graph events for Neo4j integration
            
        Returns:
            Dictionary containing transaction activities and pagination info
            
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
        
        query_string = "&".join(params)
        url = f"{self.api_url}/v1/evm/activity/{wallet}"
        if query_string:
            url = f"{url}?{query_string}"
        
        try:
            # Make the API request
            response_data = self._make_request(url)
            
            # Validate response with Pydantic
            validated_response = ActivityResponse(**response_data)
            
            # Emit graph events if requested
            if emit_graph_events and validated_response.activity:
                self._emit_graph_events(wallet, validated_response.activity)
            
            # Return the validated response as a dictionary
            return validated_response.dict()
            
        except Exception as e:
            logger.error(f"Error fetching activity for wallet {wallet}: {str(e)}")
            raise
    
    def _emit_graph_events(self, wallet: str, activities: List[ActivityItem]) -> None:
        """
        Emit graph events for Neo4j integration.
        
        Args:
            wallet: The wallet address
            activities: List of transaction activities
        """
        try:
            # Process each activity and prepare graph data
            for activity in activities:
                # Determine edge type based on activity type
                edge_type = activity.type.upper()  # SEND, RECEIVE, CALL, etc.
                
                # Prepare data for Neo4j
                graph_data = {
                    "wallet_address": wallet,
                    "activity": activity.dict(),
                    "edge_type": edge_type,
                    "timestamp": int(time.time())
                }
                
                # For transfers, include from/to addresses
                if activity.type in ['send', 'receive']:
                    graph_data["from_address"] = activity.from_address
                    graph_data["to_address"] = activity.to_address
                    graph_data["token_address"] = activity.token_address
                    graph_data["amount"] = activity.amount
                    graph_data["value_usd"] = activity.value_usd
                
                # For contract interactions, include contract address and function
                if activity.type == 'call' and activity.function:
                    graph_data["contract_address"] = activity.to_address
                    graph_data["function_name"] = activity.function.name
                    if activity.function.parameters:
                        graph_data["function_params"] = [
                            {
                                "name": param.name,
                                "type": param.type,
                                "value": param.value
                            }
                            for param in activity.function.parameters
                        ]
                
                # Emit event for graph processing
                emit_event(
                    GraphAddEvent(
                        type="wallet_activity",
                        data=graph_data
                    )
                )
            
            logger.debug(f"Emitted graph events for wallet {wallet} with {len(activities)} activities")
        except Exception as e:
            logger.error(f"Failed to emit graph events: {str(e)}")
            # Don't re-raise, as this is a non-critical operation
