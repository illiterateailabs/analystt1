"""
Sim API Client

This module provides a client for interacting with Sim APIs, which offer
real-time blockchain data across 60+ EVM chains and Solana. The client
handles authentication, request construction, error handling, and pagination.

Usage:
    sim_client = SimClient()
    balances = await sim_client.get_balances("0xd8da6bf26964af9d7eed9e03e53415d37aa96045")
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union, TypeVar, Generic, cast
import asyncio
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field, validator

from backend.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic pagination
T = TypeVar('T')

# Constants
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 1.5
DEFAULT_BACKOFF_MAX = 60  # seconds


class SimApiError(Exception):
    """Base exception for Sim API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)
    
    def __str__(self) -> str:
        if self.status_code:
            return f"SimApiError ({self.status_code}): {self.message}"
        return f"SimApiError: {self.message}"


class SimRateLimitError(SimApiError):
    """Exception raised when rate limits are exceeded."""
    pass


class SimAuthError(SimApiError):
    """Exception raised for authentication errors."""
    pass


class SimNotFoundError(SimApiError):
    """Exception raised when a resource is not found."""
    pass


class SimValidationError(SimApiError):
    """Exception raised for validation errors."""
    pass


class SimServerError(SimApiError):
    """Exception raised for server-side errors."""
    pass


class PaginatedResponse(Generic[T], BaseModel):
    """Generic model for paginated responses."""
    
    items: List[T]
    next_offset: Optional[str] = None
    total_count: Optional[int] = None
    request_time: Optional[str] = None
    response_time: Optional[str] = None


class SimClient:
    """
    Client for interacting with Sim APIs.
    
    This client provides methods for accessing real-time blockchain data
    across multiple EVM chains and Solana. It handles authentication,
    request construction, error handling, and pagination.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    ):
        """
        Initialize the Sim API client.
        
        Args:
            api_key: Sim API key (defaults to settings.SIM_API_KEY)
            api_url: Sim API base URL (defaults to settings.SIM_API_URL)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff factor for retries
        """
        self.api_key = api_key or settings.SIM_API_KEY
        self.api_url = api_url or settings.SIM_API_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Validate API key
        if not self.api_key:
            logger.warning("No Sim API key provided. API requests will fail.")
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "X-Sim-Api-Key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": f"AnalystAgent/{settings.VERSION}"
            }
        )
        
        logger.info(f"SimClient initialized with API URL: {self.api_url}")
    
    async def close(self) -> None:
        """Close the HTTP client session."""
        await self.client.aclose()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a request to the Sim API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON request body
            **kwargs: Additional parameters to pass to the request
            
        Returns:
            API response as a dictionary
            
        Raises:
            SimApiError: If the request fails after retries
        """
        url = urljoin(self.api_url, endpoint)
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                logger.debug(f"Sending {method} request to {url}")
                response = await self.client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    **kwargs
                )
                
                # Log request metrics
                logger.debug(f"Sim API request: {method} {url} - Status: {response.status_code}")
                
                # Handle successful response
                if response.status_code == 200:
                    return response.json()
                
                # Handle error responses
                error_body = response.text
                try:
                    error_json = response.json()
                    error_message = error_json.get("error", {}).get("message", "Unknown error")
                except:
                    error_message = error_body[:100] + "..." if len(error_body) > 100 else error_body
                
                # Handle specific error codes
                if response.status_code == 401:
                    raise SimAuthError("Authentication failed. Check your API key.", response.status_code, error_body)
                elif response.status_code == 404:
                    raise SimNotFoundError(f"Resource not found: {endpoint}", response.status_code, error_body)
                elif response.status_code == 422:
                    raise SimValidationError(f"Validation error: {error_message}", response.status_code, error_body)
                elif response.status_code == 429:
                    # Rate limit exceeded, apply backoff
                    retry_after = int(response.headers.get("Retry-After", "1"))
                    wait_time = min(retry_after, self.backoff_factor * (2 ** retries))
                    logger.warning(f"Rate limit exceeded. Retrying after {wait_time} seconds.")
                    await asyncio.sleep(wait_time)
                    retries += 1
                    last_error = SimRateLimitError(
                        f"Rate limit exceeded: {error_message}", 
                        response.status_code, 
                        error_body
                    )
                    continue
                elif 500 <= response.status_code < 600:
                    # Server error, apply backoff and retry
                    wait_time = self.backoff_factor * (2 ** retries)
                    logger.warning(f"Server error ({response.status_code}). Retrying after {wait_time} seconds.")
                    await asyncio.sleep(wait_time)
                    retries += 1
                    last_error = SimServerError(
                        f"Server error: {error_message}", 
                        response.status_code, 
                        error_body
                    )
                    continue
                else:
                    # Other errors
                    raise SimApiError(f"API error: {error_message}", response.status_code, error_body)
                    
            except (httpx.RequestError, httpx.TimeoutException) as e:
                # Network or timeout error, apply backoff and retry
                wait_time = self.backoff_factor * (2 ** retries)
                logger.warning(f"Request error: {str(e)}. Retrying after {wait_time} seconds.")
                await asyncio.sleep(wait_time)
                retries += 1
                last_error = SimApiError(f"Request failed: {str(e)}")
                continue
                
            except (SimAuthError, SimNotFoundError, SimValidationError, SimApiError) as e:
                # Don't retry these errors
                raise
        
        # If we've exhausted retries, raise the last error
        if last_error:
            raise last_error
        else:
            raise SimApiError("Maximum retries exceeded with no specific error")
    
    async def paged_get(
        self,
        endpoint: str,
        items_key: str,
        limit: int = 100,
        max_items: Optional[int] = None,
        **params
    ) -> List[Dict[str, Any]]:
        """
        Handle paginated GET requests with cursor-based pagination.
        
        Args:
            endpoint: API endpoint path
            items_key: Key in the response that contains the items list
            limit: Number of items per page
            max_items: Maximum total items to retrieve (None for all)
            **params: Additional query parameters
            
        Returns:
            Combined list of items from all pages
        """
        all_items = []
        params = {**params, "limit": min(limit, 100)}  # Ensure limit is reasonable
        offset = None
        
        while True:
            # Add offset parameter if we have one
            if offset:
                params["offset"] = offset
            
            # Make the request
            response = await self._request("GET", endpoint, params=params)
            
            # Extract items
            items = response.get(items_key, [])
            if not items:
                break
                
            all_items.extend(items)
            
            # Check if we've reached the maximum items
            if max_items and len(all_items) >= max_items:
                all_items = all_items[:max_items]
                break
                
            # Get next offset for pagination
            offset = response.get("next_offset")
            if not offset:
                break
                
            # Log progress
            logger.debug(f"Retrieved {len(items)} items, total: {len(all_items)}, fetching next page...")
        
        return all_items
    
    async def get_balances(
        self,
        wallet: str,
        limit: int = 100,
        offset: Optional[str] = None,
        chain_ids: str = "all",
        metadata: str = "url,logo"
    ) -> Dict[str, Any]:
        """
        Get token balances for a wallet address.
        
        Args:
            wallet: Wallet address
            limit: Maximum number of balances to return
            offset: Pagination offset token
            chain_ids: Comma-separated list of chain IDs or "all"
            metadata: Comma-separated list of metadata to include
            
        Returns:
            Dictionary containing token balances and pagination info
        """
        params = {
            "limit": limit,
            "chain_ids": chain_ids,
            "metadata": metadata
        }
        
        if offset:
            params["offset"] = offset
        
        endpoint = f"/v1/evm/balances/{wallet}"
        return await self._request("GET", endpoint, params=params)
    
    async def get_activity(
        self,
        wallet: str,
        limit: int = 25,
        offset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get chronological activity for a wallet address.
        
        Args:
            wallet: Wallet address
            limit: Maximum number of activities to return
            offset: Pagination offset token
            
        Returns:
            Dictionary containing activity data and pagination info
        """
        params = {"limit": limit}
        
        if offset:
            params["offset"] = offset
        
        endpoint = f"/v1/evm/activity/{wallet}"
        return await self._request("GET", endpoint, params=params)
    
    async def get_collectibles(
        self,
        wallet: str,
        limit: int = 50,
        offset: Optional[str] = None,
        chain_ids: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get NFT collectibles for a wallet address.
        
        Args:
            wallet: Wallet address
            limit: Maximum number of collectibles to return
            offset: Pagination offset token
            chain_ids: Comma-separated list of chain IDs or "all"
            
        Returns:
            Dictionary containing collectibles data and pagination info
        """
        params = {"limit": limit}
        
        if offset:
            params["offset"] = offset
        
        if chain_ids:
            params["chain_ids"] = chain_ids
        
        endpoint = f"/v1/evm/collectibles/{wallet}"
        return await self._request("GET", endpoint, params=params)
    
    async def get_token_info(
        self,
        token_address: str,
        chain_ids: str
    ) -> Dict[str, Any]:
        """
        Get detailed token metadata and pricing.
        
        Args:
            token_address: Token contract address or "native"
            chain_ids: Comma-separated list of chain IDs (required)
            
        Returns:
            Dictionary containing token information
        """
        params = {"chain_ids": chain_ids}
        endpoint = f"/v1/evm/token-info/{token_address}"
        return await self._request("GET", endpoint, params=params)
    
    async def get_token_holders(
        self,
        chain_id: str,
        token_address: str,
        limit: int = 100,
        offset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get token holder distribution.
        
        Args:
            chain_id: Chain ID
            token_address: Token contract address
            limit: Maximum number of holders to return
            offset: Pagination offset token
            
        Returns:
            Dictionary containing token holder data
        """
        params = {"limit": limit}
        
        if offset:
            params["offset"] = offset
        
        endpoint = f"/v1/evm/token-holders/{chain_id}/{token_address}"
        return await self._request("GET", endpoint, params=params)
    
    async def get_transactions(
        self,
        wallet: str,
        limit: int = 25,
        offset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed transaction information for a wallet.
        
        Args:
            wallet: Wallet address
            limit: Maximum number of transactions to return
            offset: Pagination offset token
            
        Returns:
            Dictionary containing transaction data
        """
        params = {"limit": limit}
        
        if offset:
            params["offset"] = offset
        
        endpoint = f"/v1/evm/transactions/{wallet}"
        return await self._request("GET", endpoint, params=params)
    
    async def get_supported_chains(self) -> Dict[str, Any]:
        """
        Get list of all supported EVM chains and their capabilities.
        
        Returns:
            Dictionary containing supported chains information
        """
        endpoint = "/v1/evm/supported-chains"
        return await self._request("GET", endpoint)
    
    async def get_svm_balances(
        self,
        wallet: str,
        limit: int = 100,
        offset: Optional[str] = None,
        chains: str = "all"
    ) -> Dict[str, Any]:
        """
        Get Solana (SVM) token balances for a wallet address.
        
        Args:
            wallet: Solana wallet address
            limit: Maximum number of balances to return
            offset: Pagination offset token
            chains: Comma-separated list of chains or "all"
            
        Returns:
            Dictionary containing SVM token balances
        """
        params = {
            "limit": limit,
            "chains": chains
        }
        
        if offset:
            params["offset"] = offset
        
        endpoint = f"/beta/svm/balances/{wallet}"
        return await self._request("GET", endpoint, params=params)
    
    async def get_svm_token_metadata(self, mint: str) -> Dict[str, Any]:
        """
        Get metadata for a Solana token mint address.
        
        Args:
            mint: Solana token mint address
            
        Returns:
            Dictionary containing token metadata
        """
        endpoint = f"/beta/svm/token-metadata/{mint}"
        return await self._request("GET", endpoint)
    
    async def get_all_balances(
        self,
        wallet: str,
        max_items: Optional[int] = None,
        chain_ids: str = "all",
        metadata: str = "url,logo"
    ) -> List[Dict[str, Any]]:
        """
        Get all token balances for a wallet with automatic pagination.
        
        Args:
            wallet: Wallet address
            max_items: Maximum number of balances to retrieve (None for all)
            chain_ids: Comma-separated list of chain IDs or "all"
            metadata: Comma-separated list of metadata to include
            
        Returns:
            List of token balances
        """
        return await self.paged_get(
            f"/v1/evm/balances/{wallet}",
            "balances",
            max_items=max_items,
            chain_ids=chain_ids,
            metadata=metadata
        )
    
    async def get_all_activity(
        self,
        wallet: str,
        max_items: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all activity for a wallet with automatic pagination.
        
        Args:
            wallet: Wallet address
            max_items: Maximum number of activities to retrieve (None for all)
            
        Returns:
            List of activity items
        """
        return await self.paged_get(
            f"/v1/evm/activity/{wallet}",
            "activity",
            max_items=max_items
        )
    
    async def get_all_collectibles(
        self,
        wallet: str,
        max_items: Optional[int] = None,
        chain_ids: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all NFT collectibles for a wallet with automatic pagination.
        
        Args:
            wallet: Wallet address
            max_items: Maximum number of collectibles to retrieve (None for all)
            chain_ids: Comma-separated list of chain IDs or "all"
            
        Returns:
            List of collectibles
        """
        params = {}
        if chain_ids:
            params["chain_ids"] = chain_ids
            
        return await self.paged_get(
            f"/v1/evm/collectibles/{wallet}",
            "entries",
            max_items=max_items,
            **params
        )
    
    async def get_all_token_holders(
        self,
        chain_id: str,
        token_address: str,
        max_items: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all token holders with automatic pagination.
        
        Args:
            chain_id: Chain ID
            token_address: Token contract address
            max_items: Maximum number of holders to retrieve (None for all)
            
        Returns:
            List of token holders
        """
        return await self.paged_get(
            f"/v1/evm/token-holders/{chain_id}/{token_address}",
            "holders",
            max_items=max_items
        )
    
    async def get_all_transactions(
        self,
        wallet: str,
        max_items: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all transactions for a wallet with automatic pagination.
        
        Args:
            wallet: Wallet address
            max_items: Maximum number of transactions to retrieve (None for all)
            
        Returns:
            List of transactions
        """
        return await self.paged_get(
            f"/v1/evm/transactions/{wallet}",
            "transactions",
            max_items=max_items
        )
    
    async def get_all_svm_balances(
        self,
        wallet: str,
        max_items: Optional[int] = None,
        chains: str = "all"
    ) -> List[Dict[str, Any]]:
        """
        Get all Solana token balances with automatic pagination.
        
        Args:
            wallet: Solana wallet address
            max_items: Maximum number of balances to retrieve (None for all)
            chains: Comma-separated list of chains or "all"
            
        Returns:
            List of SVM token balances
        """
        return await self.paged_get(
            f"/beta/svm/balances/{wallet}",
            "balances",
            max_items=max_items,
            chains=chains
        )
