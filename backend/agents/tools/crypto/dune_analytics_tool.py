"""
Dune Analytics API integration for blockchain data analysis.

This module provides a tool for interacting with Dune Analytics API v1,
allowing agents to execute SQL queries against blockchain data, retrieve
results, and store them in Neo4j for further analysis.

Documentation: https://dune.com/docs/api/
"""

import logging
import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin

import aiohttp
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.config import settings
from backend.integrations.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

class DuneAPIConfig:
    """Configuration for Dune Analytics API."""
    
    # Base URL for Dune API v1
    BASE_URL = "https://api.dune.com/api/v1/"
    
    # API Key from settings or environment
    API_KEY = getattr(settings, "dune_api_key", None)
    
    # Rate limiting settings
    MAX_REQUESTS_PER_MINUTE = 40  # Free tier limit
    REQUEST_TIMEOUT = 30  # seconds
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_MIN_WAIT = 2  # seconds
    RETRY_MAX_WAIT = 10  # seconds
    
    # Query execution settings
    MAX_WAIT_TIME = 300  # seconds to wait for query execution
    POLL_INTERVAL = 2  # seconds between status checks

class DuneQueryStatus:
    """Enum-like class for Dune query execution statuses."""
    PENDING = "QUERY_STATE_PENDING"
    EXECUTING = "QUERY_STATE_EXECUTING"
    COMPLETED = "QUERY_STATE_COMPLETED"
    FAILED = "QUERY_STATE_FAILED"
    CANCELLED = "QUERY_STATE_CANCELLED"
    EXPIRED = "QUERY_STATE_EXPIRED"

class DuneAPIError(Exception):
    """Exception for Dune API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

class DuneAnalyticsTool:
    """Tool for interacting with Dune Analytics API for blockchain data analysis."""
    
    def __init__(self, api_key: Optional[str] = None, neo4j_client: Optional[Neo4jClient] = None):
        """
        Initialize the Dune Analytics tool.
        
        Args:
            api_key: Dune API key (optional, falls back to config)
            neo4j_client: Neo4j client for storing results (optional)
        """
        self.config = DuneAPIConfig()
        self.api_key = api_key or self.config.API_KEY
        
        if not self.api_key:
            logger.warning("No Dune API key provided. Tool will not function without an API key.")
        
        self.neo4j_client = neo4j_client
        self._session = None
        self._request_timestamps = []
        
        # Tool metadata for CrewAI
        self.name = "dune_analytics_tool"
        self.description = "Execute SQL queries against blockchain data using Dune Analytics"
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self):
        """Connect to Dune API by creating an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"X-Dune-API-Key": self.api_key}
            )
            logger.debug("Created new aiohttp session for Dune API")
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("Closed aiohttp session for Dune API")
    
    async def _enforce_rate_limit(self):
        """
        Enforce rate limiting based on the configured limits.
        
        This method keeps track of request timestamps and delays if needed
        to stay within the rate limits.
        """
        now = time.time()
        
        # Remove timestamps older than 60 seconds
        self._request_timestamps = [ts for ts in self._request_timestamps if now - ts < 60]
        
        # If we've hit the limit, wait until we can make another request
        if len(self._request_timestamps) >= self.config.MAX_REQUESTS_PER_MINUTE:
            oldest = min(self._request_timestamps)
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds before next request.")
                await asyncio.sleep(wait_time)
        
        # Add current timestamp to the list
        self._request_timestamps.append(time.time())
    
    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(DuneAPIConfig.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=DuneAPIConfig.RETRY_MIN_WAIT,
            max=DuneAPIConfig.RETRY_MAX_WAIT
        )
    )
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Make a request to the Dune API with rate limiting and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for the request
        
        Returns:
            Response data as dictionary
        
        Raises:
            DuneAPIError: If the request fails
        """
        await self.connect()  # Ensure session exists
        await self._enforce_rate_limit()  # Apply rate limiting
        
        url = urljoin(self.config.BASE_URL, endpoint)
        
        try:
            async with self._session.request(
                method=method,
                url=url,
                timeout=self.config.REQUEST_TIMEOUT,
                **kwargs
            ) as response:
                response_text = await response.text()
                
                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError:
                    raise DuneAPIError(
                        f"Invalid JSON response: {response_text[:100]}...",
                        status_code=response.status
                    )
                
                if response.status >= 400:
                    error_message = data.get("error", {}).get("message", "Unknown error")
                    raise DuneAPIError(
                        f"Dune API error: {error_message}",
                        status_code=response.status,
                        response=data
                    )
                
                return data
        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during Dune API request: {str(e)}")
            raise DuneAPIError(f"HTTP error: {str(e)}")
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout during Dune API request to {endpoint}")
            raise DuneAPIError("Request timed out")
    
    async def execute_query(
        self,
        query_id: int,
        parameters: Optional[Dict[str, Any]] = None,
        wait_for_completion: bool = True,
        max_wait_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute an existing Dune query with optional parameters.
        
        Args:
            query_id: ID of the existing Dune query
            parameters: Dictionary of parameters for the query
            wait_for_completion: Whether to wait for query completion
            max_wait_time: Maximum time to wait in seconds
        
        Returns:
            Dictionary with execution details and results if completed
        
        Raises:
            DuneAPIError: If execution fails
        """
        logger.info(f"Executing Dune query ID: {query_id}")
        
        # Prepare parameters in the format Dune expects
        formatted_params = {}
        if parameters:
            for key, value in parameters.items():
                # Remove any leading colon from parameter names
                param_name = key[1:] if key.startswith(":") else key
                formatted_params[param_name] = value
        
        # Execute the query
        execution_data = await self._make_request(
            method="POST",
            endpoint=f"query/{query_id}/execute",
            json={"parameters": formatted_params} if formatted_params else {}
        )
        
        execution_id = execution_data.get("execution_id")
        if not execution_id:
            raise DuneAPIError("No execution ID returned from Dune API")
        
        logger.debug(f"Query execution initiated with ID: {execution_id}")
        
        # Return immediately if not waiting for completion
        if not wait_for_completion:
            return execution_data
        
        # Wait for query completion
        return await self._wait_for_query_completion(
            execution_id, 
            max_wait_time or self.config.MAX_WAIT_TIME
        )
    
    async def create_custom_query(
        self,
        name: str,
        raw_sql: str,
        parameters: Optional[Dict[str, Any]] = None,
        wait_for_completion: bool = True,
        max_wait_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create and execute a custom SQL query.
        
        Note: This is a premium feature in Dune and may not be available
        on free tier accounts.
        
        Args:
            name: Name for the custom query
            raw_sql: SQL query text
            parameters: Dictionary of parameters for the query
            wait_for_completion: Whether to wait for query completion
            max_wait_time: Maximum time to wait in seconds
        
        Returns:
            Dictionary with execution details and results if completed
        
        Raises:
            DuneAPIError: If creation or execution fails
        """
        logger.info(f"Creating custom Dune query: {name}")
        
        # This endpoint may not be available on free tier
        try:
            query_data = await self._make_request(
                method="POST",
                endpoint="query/",
                json={
                    "name": name,
                    "raw_sql": raw_sql,
                    "parameters": parameters or {}
                }
            )
            
            query_id = query_data.get("id")
            if not query_id:
                raise DuneAPIError("No query ID returned from Dune API")
            
            logger.debug(f"Custom query created with ID: {query_id}")
            
            # Execute the newly created query
            return await self.execute_query(
                query_id=query_id,
                parameters=parameters,
                wait_for_completion=wait_for_completion,
                max_wait_time=max_wait_time
            )
            
        except DuneAPIError as e:
            if e.status_code == 403:
                logger.error("Custom query creation failed: This feature may require a paid Dune subscription")
            raise
    
    async def _wait_for_query_completion(
        self,
        execution_id: str,
        max_wait_time: int
    ) -> Dict[str, Any]:
        """
        Wait for a query execution to complete.
        
        Args:
            execution_id: Execution ID to check
            max_wait_time: Maximum time to wait in seconds
        
        Returns:
            Dictionary with execution details and results
        
        Raises:
            DuneAPIError: If execution fails or times out
        """
        start_time = time.time()
        poll_interval = self.config.POLL_INTERVAL
        
        while time.time() - start_time < max_wait_time:
            # Get execution status
            status_data = await self._make_request(
                method="GET",
                endpoint=f"execution/{execution_id}/status"
            )
            
            state = status_data.get("state")
            
            if state == DuneQueryStatus.COMPLETED:
                logger.info(f"Query execution completed: {execution_id}")
                # Get the results
                return await self.get_query_results(execution_id)
            
            elif state in (DuneQueryStatus.FAILED, DuneQueryStatus.CANCELLED, DuneQueryStatus.EXPIRED):
                error_message = status_data.get("error", {}).get("message", "Unknown error")
                raise DuneAPIError(f"Query execution failed: {error_message}", response=status_data)
            
            # Still pending or executing, wait and try again
            logger.debug(f"Query execution in progress: {state}")
            await asyncio.sleep(poll_interval)
            
            # Increase poll interval gradually to avoid too many requests
            poll_interval = min(poll_interval * 1.5, 10)
        
        # If we get here, we've timed out
        raise DuneAPIError(f"Query execution timed out after {max_wait_time} seconds")
    
    async def get_query_results(self, execution_id: str) -> Dict[str, Any]:
        """
        Get results for a completed query execution.
        
        Args:
            execution_id: Execution ID to get results for
        
        Returns:
            Dictionary with execution details and results
        
        Raises:
            DuneAPIError: If results cannot be retrieved
        """
        logger.info(f"Getting results for execution: {execution_id}")
        
        results_data = await self._make_request(
            method="GET",
            endpoint=f"execution/{execution_id}/results"
        )
        
        return results_data
    
    async def get_query_metadata(self, query_id: int) -> Dict[str, Any]:
        """
        Get metadata for a query.
        
        Args:
            query_id: ID of the query
        
        Returns:
            Dictionary with query metadata
        """
        logger.info(f"Getting metadata for query: {query_id}")
        
        metadata = await self._make_request(
            method="GET",
            endpoint=f"query/{query_id}"
        )
        
        return metadata
    
    async def search_queries(
        self,
        term: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search for public queries on Dune.
        
        Args:
            term: Search term
            limit: Maximum number of results to return
            offset: Offset for pagination
        
        Returns:
            List of query metadata dictionaries
        """
        logger.info(f"Searching for queries with term: {term}")
        
        # Note: This endpoint may not be available or may require different
        # authentication. This is a placeholder implementation.
        try:
            search_results = await self._make_request(
                method="GET",
                endpoint="queries/search",
                params={"term": term, "limit": limit, "offset": offset}
            )
            
            return search_results.get("queries", [])
        except DuneAPIError as e:
            logger.warning(f"Query search failed: {e}")
            return []
    
    async def store_results_in_neo4j(
        self,
        results: Dict[str, Any],
        query_name: str,
        node_label: str = "DuneResult",
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Store query results in Neo4j for further analysis.
        
        Args:
            results: Query results from Dune
            query_name: Name of the query (for reference)
            node_label: Neo4j node label to use
            timestamp: Timestamp for the data (defaults to now)
        
        Returns:
            Number of nodes created
        
        Raises:
            ValueError: If Neo4j client is not provided
        """
        if not self.neo4j_client:
            raise ValueError("Neo4j client is required to store results")
        
        if not self.neo4j_client.is_connected:
            await self.neo4j_client.connect()
        
        logger.info(f"Storing Dune query results in Neo4j with label: {node_label}")
        
        # Extract the actual rows from the results
        rows = results.get("result", {}).get("rows", [])
        if not rows:
            logger.warning("No results to store in Neo4j")
            return 0
        
        # Create timestamp property
        current_time = timestamp or datetime.now()
        timestamp_str = current_time.isoformat()
        
        # Store each row as a node
        created_nodes = 0
        for row in rows:
            # Add metadata properties
            row["_query_name"] = query_name
            row["_timestamp"] = timestamp_str
            
            # Create node in Neo4j
            await self.neo4j_client.create_node(
                labels=[node_label],
                properties=row
            )
            created_nodes += 1
        
        logger.info(f"Created {created_nodes} nodes in Neo4j from Dune results")
        return created_nodes
    
    async def results_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert Dune query results to a pandas DataFrame.
        
        Args:
            results: Query results from Dune
        
        Returns:
            Pandas DataFrame with the results
        """
        rows = results.get("result", {}).get("rows", [])
        if not rows:
            return pd.DataFrame()
        
        return pd.DataFrame(rows)
    
    async def results_to_dict_list(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract rows from Dune query results as a list of dictionaries.
        
        Args:
            results: Query results from Dune
        
        Returns:
            List of dictionaries, one per row
        """
        return results.get("result", {}).get("rows", [])
    
    # Crypto-specific helper methods
    
    async def analyze_wallet_activity(
        self,
        address: str,
        chain: str = "ethereum",
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze wallet activity for a specific address.
        
        Args:
            address: Wallet address to analyze
            chain: Blockchain to analyze (ethereum, polygon, etc.)
            days: Number of days to look back
        
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing wallet activity for {address} on {chain}")
        
        # Use a pre-defined query for wallet analysis
        # This is a placeholder ID - replace with actual query ID
        wallet_analysis_query_id = 1234567
        
        results = await self.execute_query(
            query_id=wallet_analysis_query_id,
            parameters={
                "address": address.lower(),
                "chain": chain,
                "days": days
            }
        )
        
        return {
            "address": address,
            "chain": chain,
            "period_days": days,
            "results": await self.results_to_dict_list(results)
        }
    
    async def track_defi_protocol(
        self,
        protocol_name: str,
        metrics: List[str] = ["tvl", "volume", "users"],
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Track key metrics for a DeFi protocol.
        
        Args:
            protocol_name: Name of the protocol (e.g., "uniswap", "aave")
            metrics: List of metrics to track
            days: Number of days to look back
        
        Returns:
            Protocol metrics data
        """
        logger.info(f"Tracking DeFi protocol: {protocol_name}")
        
        # Use a pre-defined query for protocol tracking
        # This is a placeholder ID - replace with actual query ID
        protocol_tracking_query_id = 2345678
        
        results = await self.execute_query(
            query_id=protocol_tracking_query_id,
            parameters={
                "protocol": protocol_name.lower(),
                "metrics": ",".join(metrics),
                "days": days
            }
        )
        
        return {
            "protocol": protocol_name,
            "metrics": metrics,
            "period_days": days,
            "results": await self.results_to_dict_list(results)
        }
    
    async def detect_whale_movements(
        self,
        token_address: str,
        min_amount_usd: float = 100000,
        hours: int = 24,
        chain: str = "ethereum"
    ) -> Dict[str, Any]:
        """
        Detect large token movements (whale activity).
        
        Args:
            token_address: Token contract address
            min_amount_usd: Minimum transfer amount in USD
            hours: Number of hours to look back
            chain: Blockchain to analyze
        
        Returns:
            Whale movement data
        """
        logger.info(f"Detecting whale movements for token {token_address}")
        
        # Use a pre-defined query for whale detection
        # This is a placeholder ID - replace with actual query ID
        whale_detection_query_id = 3456789
        
        results = await self.execute_query(
            query_id=whale_detection_query_id,
            parameters={
                "token": token_address.lower(),
                "min_amount_usd": min_amount_usd,
                "hours": hours,
                "chain": chain
            }
        )
        
        return {
            "token": token_address,
            "min_amount_usd": min_amount_usd,
            "period_hours": hours,
            "chain": chain,
            "movements": await self.results_to_dict_list(results)
        }
    
    async def analyze_token_holders(
        self,
        token_address: str,
        chain: str = "ethereum",
        top_n: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze token holder distribution.
        
        Args:
            token_address: Token contract address
            chain: Blockchain to analyze
            top_n: Number of top holders to analyze
        
        Returns:
            Token holder analysis
        """
        logger.info(f"Analyzing token holders for {token_address}")
        
        # Use a pre-defined query for token holder analysis
        # This is a placeholder ID - replace with actual query ID
        holder_analysis_query_id = 4567890
        
        results = await self.execute_query(
            query_id=holder_analysis_query_id,
            parameters={
                "token": token_address.lower(),
                "chain": chain,
                "top_n": top_n
            }
        )
        
        return {
            "token": token_address,
            "chain": chain,
            "top_n": top_n,
            "holders": await self.results_to_dict_list(results)
        }
    
    async def find_related_addresses(
        self,
        address: str,
        chain: str = "ethereum",
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Find addresses related to a given address through transactions.
        
        Args:
            address: Starting address
            chain: Blockchain to analyze
            max_depth: Maximum relationship depth
        
        Returns:
            Related addresses data
        """
        logger.info(f"Finding addresses related to {address}")
        
        # Use a pre-defined query for address relationships
        # This is a placeholder ID - replace with actual query ID
        address_relationship_query_id = 5678901
        
        results = await self.execute_query(
            query_id=address_relationship_query_id,
            parameters={
                "address": address.lower(),
                "chain": chain,
                "max_depth": max_depth
            }
        )
        
        # Process results into a network graph structure
        rows = await self.results_to_dict_list(results)
        
        # Create a graph structure for agent consumption
        nodes = {}
        edges = []
        
        for row in rows:
            source = row.get("source_address")
            target = row.get("target_address")
            
            if source and source not in nodes:
                nodes[source] = {"address": source, "type": row.get("source_type", "address")}
            
            if target and target not in nodes:
                nodes[target] = {"address": target, "type": row.get("target_type", "address")}
            
            if source and target:
                edges.append({
                    "source": source,
                    "target": target,
                    "value": row.get("value", 0),
                    "tx_count": row.get("tx_count", 1),
                    "first_tx": row.get("first_tx"),
                    "last_tx": row.get("last_tx")
                })
        
        return {
            "address": address,
            "chain": chain,
            "max_depth": max_depth,
            "nodes": list(nodes.values()),
            "edges": edges
        }
    
    # Tool interface methods for CrewAI
    
    async def _run(self, query_id: int, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Run the tool with the given query ID and parameters.
        This method is used by CrewAI to execute the tool.
        
        Args:
            query_id: ID of the Dune query to execute
            parameters: Parameters for the query
        
        Returns:
            String representation of the results
        """
        try:
            results = await self.execute_query(query_id=query_id, parameters=parameters)
            df = await self.results_to_dataframe(results)
            
            # Return a string representation for the agent
            if df.empty:
                return "No results found for the query."
            
            # Format results for agent consumption
            result_str = f"Query results ({len(df)} rows):\n\n"
            
            # Add column headers
            result_str += " | ".join(df.columns) + "\n"
            result_str += "-" * (sum(len(col) for col in df.columns) + 3 * (len(df.columns) - 1)) + "\n"
            
            # Add rows (limit to 20 for readability)
            max_rows = min(20, len(df))
            for _, row in df.head(max_rows).iterrows():
                result_str += " | ".join(str(val) for val in row) + "\n"
            
            if len(df) > max_rows:
                result_str += f"\n... and {len(df) - max_rows} more rows"
            
            return result_str
            
        except Exception as e:
            logger.error(f"Error executing Dune query: {str(e)}")
            return f"Error executing Dune query: {str(e)}"
