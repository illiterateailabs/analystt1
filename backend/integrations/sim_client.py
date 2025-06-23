"""
SIM Blockchain API Client

This module provides a client for interacting with the SIM (Structured Intelligence Metrics)
blockchain data API. It handles authentication, rate limiting, retries, and cost tracking
for all SIM API endpoints.

The client is configured via the central provider registry and automatically tracks
API costs for budget monitoring and back-pressure control.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import aiohttp
from aiohttp import ClientResponseError, ClientSession

from backend.core.metrics import ApiMetrics
from backend.providers import get_provider

# Configure module logger
logger = logging.getLogger(__name__)

class SimApiError(Exception):
    """Exception raised for SIM API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(message)

class SimClient:
    """
    Client for the SIM Blockchain API.
    
    This client handles authentication, rate limiting, retries, and cost tracking
    for all SIM API endpoints. It is configured via the central provider registry.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the SIM API client.
        
        Args:
            api_key: Optional API key. If not provided, it will be loaded from the provider registry.
        """
        self._provider_config = get_provider("sim")
        if not self._provider_config:
            raise ValueError("SIM provider configuration not found in registry")
        
        # Get API key from parameter, provider config, or environment variable
        self._api_key = api_key
        if not self._api_key:
            api_key_env_var = self._provider_config.get("auth", {}).get("api_key_env_var")
            if api_key_env_var:
                self._api_key = os.getenv(api_key_env_var)
        
        if not self._api_key:
            raise ValueError("SIM API key not provided and not found in environment")
        
        # Get base URL from provider config
        self._base_url = self._provider_config.get("connection_uri", "https://api.sim-blockchain.com/v1")
        
        # Get retry configuration
        retry_config = self._provider_config.get("retry_policy", {})
        self._max_retries = retry_config.get("attempts", 3)
        self._backoff_factor = retry_config.get("backoff_factor", 0.5)
        
        # Get rate limits
        self._rate_limits = self._provider_config.get("rate_limits", {})
        self._requests_per_minute = self._rate_limits.get("requests_per_minute", 60)
        
        # Get cost rules
        self._cost_rules = self._provider_config.get("cost_rules", {})
        self._default_cost = self._cost_rules.get("default_cost_per_request", 0.01)
        self._endpoint_costs = self._cost_rules.get("endpoints", {})
        
        # Initialize session
        self._session = None
        
        logger.info(f"SIM client initialized with base URL: {self._base_url}")
    
    async def _ensure_session(self) -> ClientSession:
        """
        Ensure that an aiohttp ClientSession exists.
        
        Returns:
            An aiohttp ClientSession
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "AnalystDroidOne/1.0",
                }
            )
        return self._session
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the SIM API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Optional query parameters
            data: Optional request body data
            retry_count: Current retry attempt (used internally)
            
        Returns:
            API response as a dictionary
            
        Raises:
            SimApiError: If the API request fails after all retries
        """
        session = await self._ensure_session()
        url = f"{self._base_url}/{endpoint}"
        
        try:
            # Calculate cost for this endpoint
            endpoint_name = endpoint.split("/")[0]  # Get the base endpoint name
            cost = self._endpoint_costs.get(endpoint_name, self._default_cost)
            
            # Make the request
            async with session.request(method, url, params=params, json=data) as response:
                # Check for rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", "5"))
                    logger.warning(f"Rate limited by SIM API. Retrying after {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                    return await self._make_request(method, endpoint, params, data, retry_count)
                
                # Check for other errors
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"SIM API error: {response.status} - {error_text}")
                    
                    # Retry on server errors (5xx) or specific client errors
                    if (response.status >= 500 or response.status in [408, 429]) and retry_count < self._max_retries:
                        retry_delay = self._backoff_factor * (2 ** retry_count)
                        logger.info(f"Retrying SIM API request in {retry_delay:.2f} seconds (attempt {retry_count + 1}/{self._max_retries})")
                        await asyncio.sleep(retry_delay)
                        return await self._make_request(method, endpoint, params, data, retry_count + 1)
                    
                    raise SimApiError(
                        f"SIM API request failed: {response.status}",
                        status_code=response.status,
                        response_text=error_text
                    )
                
                # Parse successful response
                result = await response.json()
                
                # Record cost after successful request
                ApiMetrics.record_api_cost("sim", endpoint_name, cost)
                
                return result
                
        except aiohttp.ClientError as e:
            # Handle network errors
            logger.error(f"Network error in SIM API request: {str(e)}")
            
            if retry_count < self._max_retries:
                retry_delay = self._backoff_factor * (2 ** retry_count)
                logger.info(f"Retrying SIM API request in {retry_delay:.2f} seconds (attempt {retry_count + 1}/{self._max_retries})")
                await asyncio.sleep(retry_delay)
                return await self._make_request(method, endpoint, params, data, retry_count + 1)
            
            raise SimApiError(f"SIM API network error after {self._max_retries} retries: {str(e)}")
    
    @ApiMetrics.track_api_call("sim", "activity")
    async def get_activity(
        self,
        address: str,
        chain_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get blockchain activity for an address.
        
        Args:
            address: Blockchain address to query
            chain_id: Optional chain ID (e.g., 'ethereum', 'bitcoin')
            limit: Maximum number of results to return
            offset: Pagination offset
            start_time: Optional start timestamp (Unix seconds)
            end_time: Optional end timestamp (Unix seconds)
            
        Returns:
            Activity data for the address
        """
        params = {
            "address": address,
            "limit": limit,
            "offset": offset
        }
        
        if chain_id:
            params["chain_id"] = chain_id
        
        if start_time:
            params["start_time"] = start_time
        
        if end_time:
            params["end_time"] = end_time
        
        return await self._make_request("GET", "activity", params=params)
    
    @ApiMetrics.track_api_call("sim", "balances")
    async def get_balances(
        self,
        address: str,
        chain_id: Optional[str] = None,
        include_tokens: bool = True,
        include_nfts: bool = False
    ) -> Dict[str, Any]:
        """
        Get token balances for an address.
        
        Args:
            address: Blockchain address to query
            chain_id: Optional chain ID (e.g., 'ethereum', 'bitcoin')
            include_tokens: Whether to include fungible tokens
            include_nfts: Whether to include non-fungible tokens
            
        Returns:
            Balance data for the address
        """
        params = {
            "address": address,
            "include_tokens": str(include_tokens).lower(),
            "include_nfts": str(include_nfts).lower()
        }
        
        if chain_id:
            params["chain_id"] = chain_id
        
        return await self._make_request("GET", "balances", params=params)
    
    @ApiMetrics.track_api_call("sim", "token-info")
    async def get_token_info(
        self,
        token_address: str,
        chain_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get information about a token.
        
        Args:
            token_address: Token contract address
            chain_id: Optional chain ID (e.g., 'ethereum', 'polygon')
            
        Returns:
            Token information
        """
        params = {
            "token_address": token_address
        }
        
        if chain_id:
            params["chain_id"] = chain_id
        
        return await self._make_request("GET", "token-info", params=params)
    
    @ApiMetrics.track_api_call("sim", "transactions")
    async def get_transactions(
        self,
        tx_hash: Optional[str] = None,
        address: Optional[str] = None,
        chain_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get transaction details.
        
        Args:
            tx_hash: Optional transaction hash to query a specific transaction
            address: Optional address to get transactions for
            chain_id: Optional chain ID (e.g., 'ethereum', 'bitcoin')
            limit: Maximum number of results to return
            offset: Pagination offset
            start_time: Optional start timestamp (Unix seconds)
            end_time: Optional end timestamp (Unix seconds)
            
        Returns:
            Transaction data
        """
        if not tx_hash and not address:
            raise ValueError("Either tx_hash or address must be provided")
        
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if tx_hash:
            params["tx_hash"] = tx_hash
        
        if address:
            params["address"] = address
        
        if chain_id:
            params["chain_id"] = chain_id
        
        if start_time:
            params["start_time"] = start_time
        
        if end_time:
            params["end_time"] = end_time
        
        return await self._make_request("GET", "transactions", params=params)
    
    @ApiMetrics.track_api_call("sim", "token-holders")
    async def get_token_holders(
        self,
        token_address: str,
        chain_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get holders of a token.
        
        Args:
            token_address: Token contract address
            chain_id: Optional chain ID (e.g., 'ethereum', 'polygon')
            limit: Maximum number of results to return
            offset: Pagination offset
            
        Returns:
            Token holder data
        """
        params = {
            "token_address": token_address,
            "limit": limit,
            "offset": offset
        }
        
        if chain_id:
            params["chain_id"] = chain_id
        
        return await self._make_request("GET", "token-holders", params=params)
    
    @ApiMetrics.track_api_call("sim", "collectibles")
    async def get_collectibles(
        self,
        address: str,
        chain_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get NFT collectibles owned by an address.
        
        Args:
            address: Blockchain address to query
            chain_id: Optional chain ID (e.g., 'ethereum', 'polygon')
            limit: Maximum number of results to return
            offset: Pagination offset
            
        Returns:
            NFT collectible data
        """
        params = {
            "address": address,
            "limit": limit,
            "offset": offset
        }
        
        if chain_id:
            params["chain_id"] = chain_id
        
        return await self._make_request("GET", "collectibles", params=params)
    
    @ApiMetrics.track_api_call("sim", "graph-data")
    async def get_graph_data(
        self,
        address: str,
        depth: int = 1,
        chain_id: Optional[str] = None,
        include_contracts: bool = True
    ) -> Dict[str, Any]:
        """
        Get graph relationship data for an address.
        
        Args:
            address: Blockchain address to query
            depth: Relationship depth to traverse (1-3)
            chain_id: Optional chain ID (e.g., 'ethereum', 'polygon')
            include_contracts: Whether to include contract interactions
            
        Returns:
            Graph relationship data
        """
        params = {
            "address": address,
            "depth": depth,
            "include_contracts": str(include_contracts).lower()
        }
        
        if chain_id:
            params["chain_id"] = chain_id
        
        return await self._make_request("GET", "graph-data", params=params)
