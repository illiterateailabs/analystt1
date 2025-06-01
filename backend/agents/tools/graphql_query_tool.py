"""
GraphQL Query Tool for Crypto Data APIs.

This module provides a tool for executing GraphQL queries against various
crypto data providers like The Graph, Dune Analytics, Bitquery, etc.
It supports query execution, subscriptions, and handles rate limiting.
"""

import asyncio
import json
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast

import httpx
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.websockets import WebsocketsTransport
from graphql import DocumentNode, parse

from backend.core.metrics import increment_counter


logger = logging.getLogger(__name__)


class GraphQLEndpoint(str, Enum):
    """Supported GraphQL endpoints."""
    THE_GRAPH = "the_graph"
    DUNE = "dune"
    BITQUERY = "bitquery"
    MESSARI = "messari"
    UNISWAP = "uniswap"
    AAVE = "aave"
    COMPOUND = "compound"
    CUSTOM = "custom"


class GraphQLQueryTool:
    """
    Tool for executing GraphQL queries against crypto data sources.
    
    Supports multiple endpoints including The Graph Protocol, Dune Analytics,
    Bitquery, and custom endpoints. Can execute queries with variables and
    supports subscriptions for real-time data.
    
    Examples:
        ```
        # Query Uniswap V3 pools via The Graph
        result = await graphql_tool.run(
            endpoint=GraphQLEndpoint.THE_GRAPH,
            query='''
                query GetUniswapPools($first: Int!, $orderBy: String!) {
                  pools(first: $first, orderBy: $orderBy, orderDirection: desc) {
                    id
                    token0 {
                      symbol
                    }
                    token1 {
                      symbol
                    }
                    volumeUSD
                    liquidity
                  }
                }
            ''',
            variables={
                "first": 10,
                "orderBy": "volumeUSD"
            },
            endpoint_url="https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
        )
        
        # Query Ethereum transactions via Bitquery
        result = await graphql_tool.run(
            endpoint=GraphQLEndpoint.BITQUERY,
            query='''
                query GetTransactions($address: String!, $limit: Int!) {
                  ethereum {
                    transactions(
                      options: {limit: $limit}
                      address: {is: $address}
                    ) {
                      hash
                      from {
                        address
                      }
                      to {
                        address
                      }
                      value
                      gasValue
                      gasPrice
                    }
                  }
                }
            ''',
            variables={
                "address": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",  # UNI token
                "limit": 5
            },
            api_key="YOUR_BITQUERY_API_KEY"
        )
        ```
    """
    
    # Default endpoints for common services
    DEFAULT_ENDPOINTS = {
        GraphQLEndpoint.THE_GRAPH: "https://api.thegraph.com/subgraphs/name/",
        GraphQLEndpoint.DUNE: "https://api.dune.com/api/v1/graphql",
        GraphQLEndpoint.BITQUERY: "https://graphql.bitquery.io",
        GraphQLEndpoint.MESSARI: "https://api.messari.io/api/v1/graphql",
        GraphQLEndpoint.UNISWAP: "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
        GraphQLEndpoint.AAVE: "https://api.thegraph.com/subgraphs/name/aave/protocol-v3",
        GraphQLEndpoint.COMPOUND: "https://api.thegraph.com/subgraphs/name/graphprotocol/compound-v2",
    }
    
    # Rate limits (requests per minute)
    RATE_LIMITS = {
        GraphQLEndpoint.THE_GRAPH: 100,
        GraphQLEndpoint.DUNE: 30,
        GraphQLEndpoint.BITQUERY: 60,
        GraphQLEndpoint.MESSARI: 60,
        GraphQLEndpoint.UNISWAP: 100,
        GraphQLEndpoint.AAVE: 100,
        GraphQLEndpoint.COMPOUND: 100,
        GraphQLEndpoint.CUSTOM: 100,  # Default for custom endpoints
    }
    
    def __init__(self):
        """Initialize the GraphQL query tool."""
        self._last_request_time: Dict[str, float] = {}
        self._clients: Dict[str, Client] = {}
        self._subscription_clients: Dict[str, Client] = {}
        self._active_subscriptions: Dict[str, asyncio.Task] = {}
    
    async def run(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        endpoint: Union[GraphQLEndpoint, str] = GraphQLEndpoint.THE_GRAPH,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 2,
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL query against the specified endpoint.
        
        Args:
            query: GraphQL query string
            variables: Variables for the GraphQL query
            endpoint: Predefined endpoint or custom
            endpoint_url: URL for the GraphQL endpoint (required for custom endpoints)
            api_key: API key for authenticated endpoints
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Dict containing the query results
            
        Raises:
            ValueError: If endpoint_url is not provided for custom endpoints
            httpx.HTTPError: For HTTP-related errors
            Exception: For other errors during query execution
        """
        start_time = time.time()
        endpoint_name = endpoint.value if isinstance(endpoint, GraphQLEndpoint) else endpoint
        
        try:
            # Validate inputs
            if endpoint == GraphQLEndpoint.CUSTOM and not endpoint_url:
                raise ValueError("endpoint_url must be provided for custom endpoints")
            
            # Get the endpoint URL
            url = endpoint_url or self.DEFAULT_ENDPOINTS.get(cast(GraphQLEndpoint, endpoint))
            if not url:
                raise ValueError(f"Unknown endpoint: {endpoint}")
            
            # Apply rate limiting
            await self._apply_rate_limit(endpoint_name)
            
            # Get or create client
            client = await self._get_client(url, api_key)
            
            # Parse the query
            parsed_query = gql(query)
            
            # Execute the query with retries
            return await self._execute_with_retries(
                client, parsed_query, variables, max_retries, retry_delay
            )
            
        except Exception as e:
            logger.error(f"GraphQL query failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "endpoint": endpoint_name,
            }
        finally:
            # Record metrics
            duration = time.time() - start_time
            increment_counter(
                "graphql_query_duration_seconds",
                value=duration,
                labels={"endpoint": endpoint_name}
            )
            increment_counter(
                "graphql_query_count",
                labels={"endpoint": endpoint_name, "success": "true" if "error" not in locals() else "false"}
            )
    
    async def subscribe(
        self,
        query: str,
        callback: callable,
        variables: Optional[Dict[str, Any]] = None,
        endpoint: Union[GraphQLEndpoint, str] = GraphQLEndpoint.THE_GRAPH,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        subscription_id: Optional[str] = None,
    ) -> str:
        """
        Create a GraphQL subscription for real-time data.
        
        Args:
            query: GraphQL subscription query
            callback: Async function to call with each result
            variables: Variables for the subscription
            endpoint: Predefined endpoint or custom
            endpoint_url: URL for the GraphQL endpoint (required for custom)
            api_key: API key for authenticated endpoints
            subscription_id: Optional ID for the subscription
            
        Returns:
            Subscription ID that can be used to cancel the subscription
            
        Raises:
            ValueError: If endpoint doesn't support subscriptions
            Exception: For other errors during subscription setup
        """
        endpoint_name = endpoint.value if isinstance(endpoint, GraphQLEndpoint) else endpoint
        
        try:
            # Generate subscription ID if not provided
            if not subscription_id:
                import uuid
                subscription_id = f"sub_{uuid.uuid4().hex[:8]}"
            
            # Validate inputs
            if endpoint == GraphQLEndpoint.CUSTOM and not endpoint_url:
                raise ValueError("endpoint_url must be provided for custom endpoints")
            
            # Get the endpoint URL
            url = endpoint_url or self.DEFAULT_ENDPOINTS.get(cast(GraphQLEndpoint, endpoint))
            if not url:
                raise ValueError(f"Unknown endpoint: {endpoint}")
            
            # Convert HTTP URL to WebSocket URL if needed
            ws_url = url.replace("http://", "ws://").replace("https://", "wss://")
            
            # Get or create subscription client
            client = await self._get_subscription_client(ws_url, api_key)
            
            # Parse the subscription query
            parsed_query = gql(query)
            
            # Start the subscription in a background task
            task = asyncio.create_task(
                self._run_subscription(client, parsed_query, variables, callback, subscription_id)
            )
            
            # Store the task
            self._active_subscriptions[subscription_id] = task
            
            return subscription_id
            
        except Exception as e:
            logger.error(f"GraphQL subscription failed: {str(e)}")
            raise
    
    async def cancel_subscription(self, subscription_id: str) -> bool:
        """
        Cancel an active subscription.
        
        Args:
            subscription_id: ID of the subscription to cancel
            
        Returns:
            True if subscription was cancelled, False if not found
        """
        if subscription_id in self._active_subscriptions:
            task = self._active_subscriptions[subscription_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._active_subscriptions[subscription_id]
            return True
        return False
    
    async def close(self):
        """Close all clients and cancel all subscriptions."""
        # Cancel all subscriptions
        for subscription_id in list(self._active_subscriptions.keys()):
            await self.cancel_subscription(subscription_id)
        
        # Close all clients
        self._clients.clear()
        self._subscription_clients.clear()
    
    async def _get_client(self, url: str, api_key: Optional[str] = None) -> Client:
        """Get or create a GraphQL client for the given URL."""
        client_key = f"{url}:{api_key or ''}"
        
        if client_key not in self._clients:
            # Set up HTTP headers
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Create transport
            transport = AIOHTTPTransport(url=url, headers=headers)
            
            # Create client
            self._clients[client_key] = Client(
                transport=transport,
                fetch_schema_from_transport=False,
            )
        
        return self._clients[client_key]
    
    async def _get_subscription_client(self, url: str, api_key: Optional[str] = None) -> Client:
        """Get or create a GraphQL subscription client for the given URL."""
        client_key = f"{url}:{api_key or ''}"
        
        if client_key not in self._subscription_clients:
            # Set up WebSocket headers
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Create transport
            transport = WebsocketsTransport(url=url, headers=headers)
            
            # Create client
            self._subscription_clients[client_key] = Client(
                transport=transport,
                fetch_schema_from_transport=False,
            )
        
        return self._subscription_clients[client_key]
    
    async def _execute_with_retries(
        self,
        client: Client,
        query: DocumentNode,
        variables: Optional[Dict[str, Any]],
        max_retries: int,
        retry_delay: int,
    ) -> Dict[str, Any]:
        """Execute a GraphQL query with retries."""
        retries = 0
        last_error = None
        
        while retries <= max_retries:
            try:
                result = await client.execute_async(
                    query,
                    variable_values=variables
                )
                return {"success": True, "data": result}
            
            except Exception as e:
                last_error = e
                retries += 1
                
                # Check if we should retry
                if retries <= max_retries:
                    # Exponential backoff
                    wait_time = retry_delay * (2 ** (retries - 1))
                    logger.warning(f"GraphQL query failed, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    break
        
        # If we get here, all retries failed
        return {
            "success": False,
            "error": str(last_error),
            "retries": retries - 1,
        }
    
    async def _run_subscription(
        self,
        client: Client,
        query: DocumentNode,
        variables: Optional[Dict[str, Any]],
        callback: callable,
        subscription_id: str,
    ):
        """Run a GraphQL subscription and call the callback for each result."""
        try:
            async for result in client.subscribe_async(
                query,
                variable_values=variables
            ):
                try:
                    await callback(result, subscription_id)
                except Exception as callback_error:
                    logger.error(f"Subscription callback error: {str(callback_error)}")
        except asyncio.CancelledError:
            # Subscription was cancelled
            logger.info(f"Subscription {subscription_id} cancelled")
            raise
        except Exception as e:
            logger.error(f"Subscription error: {str(e)}")
            # Notify callback of error
            try:
                await callback({"error": str(e)}, subscription_id)
            except Exception:
                pass
    
    async def _apply_rate_limit(self, endpoint_name: str):
        """Apply rate limiting for the endpoint."""
        now = time.time()
        
        # Get the rate limit for this endpoint
        rate_limit = self.RATE_LIMITS.get(
            cast(GraphQLEndpoint, endpoint_name),
            self.RATE_LIMITS[GraphQLEndpoint.CUSTOM]
        )
        
        # Calculate minimum time between requests
        min_interval = 60.0 / rate_limit  # seconds
        
        # Check if we need to wait
        if endpoint_name in self._last_request_time:
            elapsed = now - self._last_request_time[endpoint_name]
            if elapsed < min_interval:
                # Wait to respect rate limit
                wait_time = min_interval - elapsed
                logger.debug(f"Rate limiting {endpoint_name}, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Update last request time
        self._last_request_time[endpoint_name] = time.time()

    async def query_token_data(
        self,
        token_address: str,
        chain: str = "ethereum",
        endpoint: Union[GraphQLEndpoint, str] = GraphQLEndpoint.THE_GRAPH,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query token data for a specific token address.
        
        Args:
            token_address: Ethereum address of the token
            chain: Blockchain name (ethereum, polygon, etc.)
            endpoint: GraphQL endpoint to use
            endpoint_url: Custom endpoint URL
            api_key: API key for the endpoint
            
        Returns:
            Token data including price, volume, market cap, etc.
        """
        query = """
        query GetTokenData($address: String!) {
          token(id: $address) {
            id
            name
            symbol
            decimals
            totalSupply
            volume
            volumeUSD
            txCount
            liquidity
            derivedETH
          }
        }
        """
        
        variables = {"address": token_address.lower()}
        
        if endpoint == GraphQLEndpoint.THE_GRAPH and not endpoint_url:
            # Use appropriate subgraph based on chain
            if chain == "ethereum":
                endpoint_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
            elif chain == "polygon":
                endpoint_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-polygon"
            else:
                raise ValueError(f"Unsupported chain: {chain}")
        
        return await self.run(
            query=query,
            variables=variables,
            endpoint=endpoint,
            endpoint_url=endpoint_url,
            api_key=api_key,
        )

    async def query_wallet_transactions(
        self,
        wallet_address: str,
        limit: int = 10,
        endpoint: Union[GraphQLEndpoint, str] = GraphQLEndpoint.BITQUERY,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query transactions for a specific wallet address.
        
        Args:
            wallet_address: Ethereum wallet address
            limit: Maximum number of transactions to return
            endpoint: GraphQL endpoint to use
            api_key: API key for the endpoint
            
        Returns:
            Recent transactions for the wallet
        """
        query = """
        query GetWalletTransactions($address: String!, $limit: Int!) {
          ethereum {
            transactions(
              options: {limit: $limit, desc: "block.timestamp"}
              address: {is: $address}
            ) {
              hash
              block {
                timestamp {
                  time
                }
                height
              }
              from {
                address
              }
              to {
                address
              }
              value
              gasValue
              gasPrice
            }
          }
        }
        """
        
        variables = {
            "address": wallet_address,
            "limit": limit
        }
        
        return await self.run(
            query=query,
            variables=variables,
            endpoint=endpoint,
            api_key=api_key,
        )

    async def query_defi_protocol_stats(
        self,
        protocol: str = "uniswap",
        days: int = 7,
        endpoint: Union[GraphQLEndpoint, str] = GraphQLEndpoint.THE_GRAPH,
    ) -> Dict[str, Any]:
        """
        Query statistics for a DeFi protocol.
        
        Args:
            protocol: Protocol name (uniswap, aave, compound, etc.)
            days: Number of days of data to return
            endpoint: GraphQL endpoint to use
            
        Returns:
            Protocol statistics including TVL, volume, fees, etc.
        """
        if protocol.lower() == "uniswap":
            endpoint_url = self.DEFAULT_ENDPOINTS[GraphQLEndpoint.UNISWAP]
            query = """
            query GetUniswapStats($days: Int!) {
              uniswapDayDatas(first: $days, orderBy: date, orderDirection: desc) {
                date
                volumeUSD
                tvlUSD
                feesUSD
              }
            }
            """
        elif protocol.lower() == "aave":
            endpoint_url = self.DEFAULT_ENDPOINTS[GraphQLEndpoint.AAVE]
            query = """
            query GetAaveStats($days: Int!) {
              marketDailySnapshots(first: $days, orderBy: timestamp, orderDirection: desc) {
                timestamp
                totalValueLockedUSD
                totalBorrowBalanceUSD
                totalDepositBalanceUSD
                dailySupplySideRevenueUSD
                dailyProtocolSideRevenueUSD
              }
            }
            """
        elif protocol.lower() == "compound":
            endpoint_url = self.DEFAULT_ENDPOINTS[GraphQLEndpoint.COMPOUND]
            query = """
            query GetCompoundStats($days: Int!) {
              marketDailySnapshots(first: $days, orderBy: timestamp, orderDirection: desc) {
                timestamp
                totalValueLockedUSD
                totalBorrowBalanceUSD
                totalDepositBalanceUSD
                dailySupplySideRevenueUSD
                dailyProtocolSideRevenueUSD
              }
            }
            """
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
        
        variables = {"days": days}
        
        return await self.run(
            query=query,
            variables=variables,
            endpoint=endpoint,
            endpoint_url=endpoint_url,
        )
