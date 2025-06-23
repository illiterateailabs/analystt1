"""
SIM API Client for Blockchain Data

This module provides a robust and comprehensive client for interacting with the 
SIM Blockchain Data API. It is designed for high-performance, asynchronous
operations and is fully integrated with the application's core systems.

Key Features:
- Asynchronous API calls using httpx.
- Integration with the BackpressureMiddleware for rate limiting, budget control,
  and circuit breaking.
- Automated cost tracking that emits metrics based on the provider registry.
- Centralized configuration loading from the provider registry.
- Graceful error handling and detailed logging.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException, status

from backend.core.backpressure import with_backpressure
from backend.core.metrics import ApiMetrics
from backend.providers import get_provider

logger = logging.getLogger(__name__)


class SimApiClient:
    """
    An asynchronous client for the SIM Blockchain Data API.
    """

    def __init__(self):
        """
        Initializes the SimApiClient.

        Loads configuration from the provider registry, sets up the HTTP client,
        and prepares for making authenticated API calls.
        
        Raises:
            ValueError: If the 'sim' provider is not configured in the registry
                        or if the API key is missing.
        """
        provider_config = get_provider("sim")
        if not provider_config:
            raise ValueError("SIM provider configuration not found in registry.")

        self.api_key = provider_config.get("auth", {}).get("api_key_env_var")
        if not self.api_key:
            raise ValueError("SIM_API_KEY environment variable not set.")

        self.base_url = provider_config.get("connection_uri", "https://api.sim.io/v1")
        self.cost_rules = provider_config.get("cost_rules", {})
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0  # Set a reasonable timeout for API calls
        )
        logger.info("SimApiClient initialized.")

    def _track_cost(self, endpoint: str):
        """
        Calculates and tracks the cost of an API call for a given endpoint.

        Args:
            endpoint: The name of the API endpoint that was called (e.g., "activity").
        """
        # Get cost from endpoint-specific rules, or fall back to default
        cost = self.cost_rules.get("endpoints", {}).get(
            endpoint, self.cost_rules.get("default_cost_per_request", 0.0)
        )

        if cost > 0:
            logger.debug(f"SIM API call cost: ${cost:.4f} for endpoint '{endpoint}'")
            ApiMetrics.track_credits(
                provider="sim",
                endpoint=endpoint,
                credit_type="usd",
                amount=cost
            )

    @with_backpressure(provider_id="sim", endpoint="activity")
    async def get_activity(
        self, address: str, chain: str, limit: int = 100
    ) -> Dict[str, Any]:
        """
        Fetches the recent activity for a given blockchain address.

        Args:
            address: The blockchain address.
            chain: The blockchain to query (e.g., "ethereum").
            limit: The maximum number of activity items to return.

        Returns:
            A dictionary containing the activity data.
            
        Raises:
            HTTPException: If the API call fails.
        """
        endpoint = "activity"
        params = {"address": address, "chain": chain, "limit": limit}
        logger.debug(f"Fetching SIM activity for {address} on {chain}")
        
        try:
            response = await self.client.get(f"/{endpoint}", params=params)
            response.raise_for_status()  # Raise exception for 4xx/5xx responses
            
            self._track_cost(endpoint)
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching SIM activity for {address}: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Error from SIM API: {e.response.text}"
            )
        except httpx.RequestError as e:
            logger.error(f"Request error fetching SIM activity for {address}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Could not connect to SIM API: {e}"
            )

    @with_backpressure(provider_id="sim", endpoint="balances")
    async def get_balances(self, address: str, chain: str) -> Dict[str, Any]:
        """
        Fetches the token balances for a given blockchain address.

        Args:
            address: The blockchain address.
            chain: The blockchain to query.

        Returns:
            A dictionary containing the balance data.
        """
        endpoint = "balances"
        params = {"address": address, "chain": chain}
        logger.debug(f"Fetching SIM balances for {address} on {chain}")

        try:
            response = await self.client.get(f"/{endpoint}", params=params)
            response.raise_for_status()
            
            self._track_cost(endpoint)
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching SIM balances for {address}: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Error from SIM API: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request error fetching SIM balances for {address}: {e}")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to SIM API: {e}")

    @with_backpressure(provider_id="sim", endpoint="token-info")
    async def get_token_info(self, token_address: str, chain: str) -> Dict[str, Any]:
        """
        Fetches information about a specific token.

        Args:
            token_address: The address of the token contract.
            chain: The blockchain where the token exists.

        Returns:
            A dictionary containing token metadata.
        """
        endpoint = "token-info"
        params = {"token_address": token_address, "chain": chain}
        logger.debug(f"Fetching SIM token info for {token_address} on {chain}")

        try:
            response = await self.client.get(f"/{endpoint}", params=params)
            response.raise_for_status()
            
            self._track_cost(endpoint)
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching SIM token info for {token_address}: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Error from SIM API: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request error fetching SIM token info for {token_address}: {e}")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to SIM API: {e}")

    @with_backpressure(provider_id="sim", endpoint="transactions")
    async def get_transactions(self, tx_hashes: List[str], chain: str) -> Dict[str, Any]:
        """
        Fetches details for a list of transaction hashes.

        Args:
            tx_hashes: A list of transaction hashes to query.
            chain: The blockchain to query.

        Returns:
            A dictionary containing the transaction data.
        """
        endpoint = "transactions"
        # The SIM API might expect a comma-separated string or repeated query params.
        # Assuming a POST request with a JSON body is more robust for lists.
        payload = {"tx_hashes": tx_hashes, "chain": chain}
        logger.debug(f"Fetching {len(tx_hashes)} SIM transactions on {chain}")

        try:
            response = await self.client.post(f"/{endpoint}", json=payload)
            response.raise_for_status()
            
            self._track_cost(endpoint)
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching SIM transactions: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Error from SIM API: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request error fetching SIM transactions: {e}")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to SIM API: {e}")

    async def close(self):
        """
        Closes the underlying HTTP client. Should be called on application shutdown.
        """
        await self.client.aclose()
        logger.info("SimApiClient closed.")
