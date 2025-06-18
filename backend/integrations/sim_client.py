"""
Sim API Client

A centralized client for interacting with Sim APIs, providing access to blockchain data
across 60+ EVM chains and Solana. This client handles authentication, rate limiting,
error handling, and provides methods for all Sim API endpoints.

API Reference: https://docs.sim.dune.com/
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin

import requests
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

from backend.config import settings
from backend.core.metrics import record_api_latency

# Configure logging
logger = logging.getLogger(__name__)

class SimApiError(Exception):
    """Custom exception for Sim API errors with additional context."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 error_code: Optional[str] = None, response: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self.error_code = error_code
        self.response = response
        super().__init__(message)


class SimClient:
    """
    Client for interacting with Sim APIs.
    
    This client provides methods for all Sim API endpoints, handling authentication,
    rate limiting, and error handling. It uses exponential backoff for retries and
    records metrics for API latency.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the Sim API client.
        
        Args:
            api_key: The Sim API key (defaults to settings.SIM_API_KEY)
            base_url: The base URL for Sim APIs (defaults to settings.SIM_API_URL)
        """
        self.api_key = api_key or settings.SIM_API_KEY
        self.base_url = base_url or settings.SIM_API_URL
        
        if not self.api_key:
            logger.error("Sim API key not provided. Set SIM_API_KEY in environment or pass to constructor.")
            raise ValueError("Sim API key is required")
        
        self.headers = {
            "X-Sim-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Track rate limits
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        
        logger.info(f"SimClient initialized with base URL: {self.base_url}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            requests.exceptions.RequestException,
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout
        )),
        reraise=True
    )
    def _request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the Sim API with retry logic and metrics recording.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data for POST requests
            
        Returns:
            The JSON response as a dictionary
            
        Raises:
            SimApiError: If the request fails after retries
        """
        url = urljoin(self.base_url, endpoint)
        
        # Check if we need to wait for rate limit reset
        if self.rate_limit_remaining is not None and self.rate_limit_remaining <= 1:
            if self.rate_limit_reset is not None:
                wait_time = max(0, self.rate_limit_reset - time.time())
                if wait_time > 0:
                    logger.warning(f"Rate limit almost exceeded. Waiting {wait_time:.2f}s before next request.")
                    time.sleep(wait_time)
        
        start_time = time.time()
        endpoint_name = endpoint.split('/')[2] if len(endpoint.split('/')) > 2 else endpoint
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data,
                timeout=30  # 30 second timeout
            )
            
            # Update rate limit tracking if headers are present
            if 'X-Rate-Limit-Remaining' in response.headers:
                self.rate_limit_remaining = int(response.headers['X-Rate-Limit-Remaining'])
            if 'X-Rate-Limit-Reset' in response.headers:
                self.rate_limit_reset = int(response.headers['X-Rate-Limit-Reset'])
            
            # Record API latency metric
            duration = time.time() - start_time
            record_api_latency('sim', endpoint_name, duration, response.status_code)
            
            # Handle HTTP errors
            if response.status_code >= 400:
                error_data = {}
                try:
                    error_data = response.json()
                except:
                    error_data = {"text": response.text}
                
                error_code = error_data.get('error', {}).get('code') if isinstance(error_data.get('error'), dict) else None
                error_message = error_data.get('error', {}).get('message') if isinstance(error_data.get('error'), dict) else str(error_data)
                
                message = f"Sim API error: {response.status_code} - {error_message}"
                
                # Log different levels based on status code
                if response.status_code == 429:
                    logger.warning(f"{message} (Rate limit exceeded)")
                elif response.status_code >= 500:
                    logger.error(f"{message} (Server error)")
                else:
                    logger.error(f"{message} (Client error)")
                
                raise SimApiError(
                    message=message,
                    status_code=response.status_code,
                    error_code=error_code,
                    response=error_data
                )
            
            # Parse and return JSON response
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception when calling Sim API: {str(e)}")
            # Record failure metric
            duration = time.time() - start_time
            record_api_latency('sim', endpoint_name, duration, 0)  # 0 indicates failure
            raise SimApiError(f"Request failed: {str(e)}")
    
    def get_balances(self, wallet: str, limit: int = 100, offset: Optional[str] = None,
                   chain_ids: str = "all", metadata: str = "url,logo") -> Dict[str, Any]:
        """
        Fetch token balances for a wallet address.
        
        Args:
            wallet: The wallet address to query
            limit: Maximum number of balances to return (default: 100)
            offset: Pagination offset token from previous response
            chain_ids: Comma-separated list of chain IDs or "all" for all chains
            metadata: Comma-separated list of metadata to include (url, logo)
            
        Returns:
            Dictionary containing token balances and pagination info
            
        Raises:
            SimApiError: If the request fails
        """
        endpoint = f"/v1/evm/balances/{wallet}"
        params = {}
        
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if chain_ids:
            params["chain_ids"] = chain_ids
        if metadata:
            params["metadata"] = metadata
        
        return self._request("GET", endpoint, params=params)
    
    def get_activity(self, wallet: str, limit: int = 25, offset: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch transaction activity for a wallet address.
        
        Args:
            wallet: The wallet address to query
            limit: Maximum number of activities to return (default: 25)
            offset: Pagination offset token from previous response
            
        Returns:
            Dictionary containing transaction activities and pagination info
            
        Raises:
            SimApiError: If the request fails
        """
        endpoint = f"/v1/evm/activity/{wallet}"
        params = {}
        
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        
        return self._request("GET", endpoint, params=params)
    
    def get_collectibles(self, wallet: str, limit: int = 50, offset: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch NFT collectibles for a wallet address.
        
        Args:
            wallet: The wallet address to query
            limit: Maximum number of collectibles to return (default: 50)
            offset: Pagination offset token from previous response
            
        Returns:
            Dictionary containing NFT collectibles and pagination info
            
        Raises:
            SimApiError: If the request fails
        """
        endpoint = f"/v1/evm/collectibles/{wallet}"
        params = {}
        
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        
        return self._request("GET", endpoint, params=params)
    
    def get_transactions(self, wallet: str, limit: int = 25, offset: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch detailed transaction information for a wallet address.
        
        Args:
            wallet: The wallet address to query
            limit: Maximum number of transactions to return (default: 25)
            offset: Pagination offset token from previous response
            
        Returns:
            Dictionary containing transaction details and pagination info
            
        Raises:
            SimApiError: If the request fails
        """
        endpoint = f"/v1/evm/transactions/{wallet}"
        params = {}
        
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        
        return self._request("GET", endpoint, params=params)
    
    def get_token_info(self, token_address: str, chain_ids: str = "all") -> Dict[str, Any]:
        """
        Fetch detailed metadata and pricing information for a token.
        
        Args:
            token_address: The token contract address or "native" for native tokens
            chain_ids: Comma-separated list of chain IDs or "all" for all chains
            
        Returns:
            Dictionary containing token metadata and pricing info
            
        Raises:
            SimApiError: If the request fails
        """
        endpoint = f"/v1/evm/token-info/{token_address}"
        params = {"chain_ids": chain_ids}
        
        return self._request("GET", endpoint, params=params)
    
    def get_token_holders(self, chain_id: int, token_address: str, limit: int = 100, 
                         offset: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch token holders for a specific token, ranked by wallet value.
        
        Args:
            chain_id: The chain ID where the token exists
            token_address: The token contract address
            limit: Maximum number of holders to return (default: 100)
            offset: Pagination offset token from previous response
            
        Returns:
            Dictionary containing token holders and pagination info
            
        Raises:
            SimApiError: If the request fails
        """
        endpoint = f"/v1/evm/token-holders/{chain_id}/{token_address}"
        params = {}
        
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        
        return self._request("GET", endpoint, params=params)
    
    def get_supported_chains(self) -> Dict[str, Any]:
        """
        Fetch list of all supported EVM chains and their capabilities.
        
        Returns:
            Dictionary containing supported chains information
            
        Raises:
            SimApiError: If the request fails
        """
        endpoint = "/v1/evm/supported-chains"
        return self._request("GET", endpoint)
    
    def get_svm_balances(self, wallet: str, limit: int = 100, offset: Optional[str] = None,
                        chains: str = "all") -> Dict[str, Any]:
        """
        Fetch token balances for a Solana (SVM) address.
        
        Args:
            wallet: The Solana wallet address to query
            limit: Maximum number of balances to return (default: 100)
            offset: Pagination offset token from previous response
            chains: Comma-separated list of chains or "all" for all supported chains
            
        Returns:
            Dictionary containing token balances and pagination info
            
        Raises:
            SimApiError: If the request fails
        """
        endpoint = f"/beta/svm/balances/{wallet}"
        params = {}
        
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if chains:
            params["chains"] = chains
        
        return self._request("GET", endpoint, params=params)
    
    def get_svm_transactions(self, wallet: str, limit: int = 25, offset: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch transactions for a Solana (SVM) address.
        
        Args:
            wallet: The Solana wallet address to query
            limit: Maximum number of transactions to return (default: 25)
            offset: Pagination offset token from previous response
            
        Returns:
            Dictionary containing transaction details and pagination info
            
        Raises:
            SimApiError: If the request fails
        """
        endpoint = f"/beta/svm/transactions/{wallet}"
        params = {}
        
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        
        return self._request("GET", endpoint, params=params)
    
    def get_svm_token_metadata(self, mint: str) -> Dict[str, Any]:
        """
        Fetch metadata for a Solana token mint address.
        
        Args:
            mint: The Solana token mint address
            
        Returns:
            Dictionary containing token metadata
            
        Raises:
            SimApiError: If the request fails
        """
        endpoint = f"/beta/svm/token-metadata/{mint}"
        return self._request("GET", endpoint)


# Create a singleton instance for app-wide use
sim_client = SimClient()
