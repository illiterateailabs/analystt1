"""
DefiLlama API integration for DeFi protocol analysis.

This module provides a tool for interacting with DefiLlama's APIs,
allowing agents to retrieve TVL data, yield information, protocol metrics,
and other DeFi-related data for analysis.

Documentation:
- Main API: https://defillama.com/docs/api
- Yields API: https://defillama.com/docs/api/yields
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

class DefiLlamaAPIConfig:
    """Configuration for DefiLlama APIs."""
    
    # Base URLs for different DefiLlama APIs
    MAIN_API_URL = "https://api.llama.fi/"
    YIELDS_API_URL = "https://yields.llama.fi/"
    STABLECOINS_API_URL = "https://stablecoins.llama.fi/"
    
    # Rate limiting settings (DefiLlama has generous limits for free tier)
    MAX_REQUESTS_PER_MINUTE = 200  # Free tier limit
    REQUEST_TIMEOUT = 30  # seconds
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_MIN_WAIT = 2  # seconds
    RETRY_MAX_WAIT = 10  # seconds

class DefiLlamaAPIError(Exception):
    """Exception for DefiLlama API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

class DefiLlamaTool:
    """Tool for interacting with DefiLlama APIs for DeFi protocol analysis."""
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """
        Initialize the DefiLlama tool.
        
        Args:
            neo4j_client: Neo4j client for storing results (optional)
        """
        self.config = DefiLlamaAPIConfig()
        self.neo4j_client = neo4j_client
        self._main_session = None
        self._yields_session = None
        self._stablecoins_session = None
        self._request_timestamps = []
        
        # Tool metadata for CrewAI
        self.name = "defillama_tool"
        self.description = "Analyze DeFi protocols, TVL, yields, and stablecoins using DefiLlama data"
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self):
        """Connect to DefiLlama APIs by creating aiohttp sessions."""
        if self._main_session is None or self._main_session.closed:
            self._main_session = aiohttp.ClientSession()
            logger.debug("Created new aiohttp session for DefiLlama main API")
        
        if self._yields_session is None or self._yields_session.closed:
            self._yields_session = aiohttp.ClientSession()
            logger.debug("Created new aiohttp session for DefiLlama yields API")
        
        if self._stablecoins_session is None or self._stablecoins_session.closed:
            self._stablecoins_session = aiohttp.ClientSession()
            logger.debug("Created new aiohttp session for DefiLlama stablecoins API")
    
    async def close(self):
        """Close the aiohttp sessions."""
        sessions = [
            (self._main_session, "main"),
            (self._yields_session, "yields"),
            (self._stablecoins_session, "stablecoins")
        ]
        
        for session, name in sessions:
            if session and not session.closed:
                await session.close()
                logger.debug(f"Closed aiohttp session for DefiLlama {name} API")
        
        self._main_session = None
        self._yields_session = None
        self._stablecoins_session = None
    
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
        stop=stop_after_attempt(DefiLlamaAPIConfig.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=DefiLlamaAPIConfig.RETRY_MIN_WAIT,
            max=DefiLlamaAPIConfig.RETRY_MAX_WAIT
        )
    )
    async def _make_request(
        self,
        api_type: str,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict:
        """
        Make a request to the DefiLlama API with rate limiting and error handling.
        
        Args:
            api_type: Type of API to use ('main', 'yields', 'stablecoins')
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for the request
        
        Returns:
            Response data as dictionary
        
        Raises:
            DefiLlamaAPIError: If the request fails
        """
        await self.connect()  # Ensure sessions exist
        await self._enforce_rate_limit()  # Apply rate limiting
        
        # Select the appropriate session and base URL
        if api_type == 'main':
            session = self._main_session
            base_url = self.config.MAIN_API_URL
        elif api_type == 'yields':
            session = self._yields_session
            base_url = self.config.YIELDS_API_URL
        elif api_type == 'stablecoins':
            session = self._stablecoins_session
            base_url = self.config.STABLECOINS_API_URL
        else:
            raise ValueError(f"Invalid API type: {api_type}")
        
        url = urljoin(base_url, endpoint)
        
        try:
            async with session.request(
                method=method,
                url=url,
                timeout=self.config.REQUEST_TIMEOUT,
                **kwargs
            ) as response:
                response_text = await response.text()
                
                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError:
                    raise DefiLlamaAPIError(
                        f"Invalid JSON response: {response_text[:100]}...",
                        status_code=response.status
                    )
                
                if response.status >= 400:
                    error_message = data.get("error", "Unknown error")
                    if isinstance(error_message, dict):
                        error_message = json.dumps(error_message)
                    raise DefiLlamaAPIError(
                        f"DefiLlama API error: {error_message}",
                        status_code=response.status,
                        response=data
                    )
                
                return data
        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during DefiLlama API request: {str(e)}")
            raise DefiLlamaAPIError(f"HTTP error: {str(e)}")
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout during DefiLlama API request to {endpoint}")
            raise DefiLlamaAPIError("Request timed out")
    
    # Main API Methods
    
    async def get_protocols(self) -> List[Dict[str, Any]]:
        """
        Get a list of all protocols tracked by DefiLlama.
        
        Returns:
            List of protocol data dictionaries
        """
        logger.info("Getting list of all protocols from DefiLlama")
        
        data = await self._make_request(
            api_type='main',
            method='GET',
            endpoint='protocols'
        )
        
        return data
    
    async def get_protocol_data(self, protocol_slug: str) -> Dict[str, Any]:
        """
        Get detailed data for a specific protocol.
        
        Args:
            protocol_slug: Protocol slug/name (e.g., 'aave', 'uniswap')
        
        Returns:
            Protocol data dictionary
        """
        logger.info(f"Getting data for protocol: {protocol_slug}")
        
        data = await self._make_request(
            api_type='main',
            method='GET',
            endpoint=f'protocol/{protocol_slug}'
        )
        
        return data
    
    async def get_protocol_tvl_history(
        self,
        protocol_slug: str,
        from_timestamp: Optional[int] = None,
        to_timestamp: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical TVL data for a specific protocol.
        
        Args:
            protocol_slug: Protocol slug/name (e.g., 'aave', 'uniswap')
            from_timestamp: Start timestamp (Unix timestamp in seconds)
            to_timestamp: End timestamp (Unix timestamp in seconds)
        
        Returns:
            List of TVL data points
        """
        logger.info(f"Getting TVL history for protocol: {protocol_slug}")
        
        endpoint = f'tvl/{protocol_slug}'
        params = {}
        
        if from_timestamp:
            params['from'] = from_timestamp
        if to_timestamp:
            params['to'] = to_timestamp
        
        data = await self._make_request(
            api_type='main',
            method='GET',
            endpoint=endpoint,
            params=params
        )
        
        return data
    
    async def get_chains(self) -> List[Dict[str, Any]]:
        """
        Get a list of all chains tracked by DefiLlama.
        
        Returns:
            List of chain data dictionaries
        """
        logger.info("Getting list of all chains from DefiLlama")
        
        data = await self._make_request(
            api_type='main',
            method='GET',
            endpoint='chains'
        )
        
        return data
    
    async def get_chain_tvl(self, chain: str) -> Dict[str, Any]:
        """
        Get TVL data for a specific chain.
        
        Args:
            chain: Chain name (e.g., 'ethereum', 'bsc')
        
        Returns:
            Chain TVL data dictionary
        """
        logger.info(f"Getting TVL data for chain: {chain}")
        
        data = await self._make_request(
            api_type='main',
            method='GET',
            endpoint=f'chain/{chain}'
        )
        
        return data
    
    # Yields API Methods
    
    async def get_pools(
        self,
        chain: Optional[str] = None,
        project: Optional[str] = None,
        tvl_min: Optional[int] = None,
        apy_min: Optional[float] = None,
        apy_max: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get yield pools data with optional filtering.
        
        Args:
            chain: Filter by chain (e.g., 'ethereum', 'bsc')
            project: Filter by project (e.g., 'aave', 'compound')
            tvl_min: Minimum TVL in USD
            apy_min: Minimum APY (as percentage)
            apy_max: Maximum APY (as percentage)
        
        Returns:
            List of pool data dictionaries
        """
        logger.info("Getting yield pools data from DefiLlama")
        
        params = {}
        if chain:
            params['chain'] = chain
        if project:
            params['project'] = project
        if tvl_min is not None:
            params['tvlUsd'] = f">{tvl_min}"
        if apy_min is not None:
            params['apy'] = f">{apy_min}"
        if apy_max is not None:
            params['apy'] = f"<{apy_max}" if 'apy' not in params else f"{params['apy']},{apy_max}"
        
        data = await self._make_request(
            api_type='yields',
            method='GET',
            endpoint='pools',
            params=params
        )
        
        return data
    
    async def get_pool_data(self, pool_id: str) -> Dict[str, Any]:
        """
        Get detailed data for a specific yield pool.
        
        Args:
            pool_id: Pool ID from the pools endpoint
        
        Returns:
            Pool data dictionary
        """
        logger.info(f"Getting data for pool: {pool_id}")
        
        data = await self._make_request(
            api_type='yields',
            method='GET',
            endpoint=f'pool/{pool_id}'
        )
        
        return data
    
    # Stablecoins API Methods
    
    async def get_stablecoins(self) -> Dict[str, Any]:
        """
        Get data for all stablecoins tracked by DefiLlama.
        
        Returns:
            Stablecoins data dictionary
        """
        logger.info("Getting stablecoins data from DefiLlama")
        
        data = await self._make_request(
            api_type='stablecoins',
            method='GET',
            endpoint='stablecoins'
        )
        
        return data
    
    async def get_stablecoin_charts(
        self,
        stablecoin: Optional[str] = None,
        chain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get historical charts data for stablecoins.
        
        Args:
            stablecoin: Filter by stablecoin (e.g., 'USDT', 'USDC')
            chain: Filter by chain (e.g., 'ethereum', 'bsc')
        
        Returns:
            Stablecoin charts data dictionary
        """
        logger.info("Getting stablecoin charts data from DefiLlama")
        
        params = {}
        if stablecoin:
            params['stablecoin'] = stablecoin
        if chain:
            params['chain'] = chain
        
        data = await self._make_request(
            api_type='stablecoins',
            method='GET',
            endpoint='stablecoincharts',
            params=params
        )
        
        return data
    
    # Advanced Analysis Methods
    
    async def get_protocol_tvl_breakdown(
        self,
        protocol_slug: str
    ) -> Dict[str, Any]:
        """
        Get a breakdown of TVL for a protocol by chain and token.
        
        Args:
            protocol_slug: Protocol slug/name (e.g., 'aave', 'uniswap')
        
        Returns:
            TVL breakdown dictionary
        """
        logger.info(f"Getting TVL breakdown for protocol: {protocol_slug}")
        
        protocol_data = await self.get_protocol_data(protocol_slug)
        
        # Extract TVL by chain
        tvl_by_chain = {}
        for chain in protocol_data.get('chainTvls', {}):
            # Skip special keys like 'tvl'
            if chain in ['tvl']:
                continue
            tvl_by_chain[chain] = protocol_data['chainTvls'][chain].get('tvl', 0)
        
        # Extract token breakdown if available
        tokens = protocol_data.get('tokens', [])
        token_breakdown = {}
        for token_data in tokens:
            chain = token_data.get('chain', 'unknown')
            token = token_data.get('symbol', token_data.get('name', 'unknown'))
            amount = token_data.get('amount', 0)
            price = token_data.get('price', 0)
            value = token_data.get('value', 0)
            
            if chain not in token_breakdown:
                token_breakdown[chain] = []
            
            token_breakdown[chain].append({
                'token': token,
                'amount': amount,
                'price': price,
                'value': value
            })
        
        return {
            'protocol': protocol_slug,
            'total_tvl': protocol_data.get('tvl', 0),
            'tvl_by_chain': tvl_by_chain,
            'token_breakdown': token_breakdown
        }
    
    async def compare_protocols(
        self,
        protocol_slugs: List[str],
        metrics: List[str] = ['tvl', 'mcap', 'fdv']
    ) -> Dict[str, Any]:
        """
        Compare multiple protocols across selected metrics.
        
        Args:
            protocol_slugs: List of protocol slugs to compare
            metrics: List of metrics to compare (tvl, mcap, fdv)
        
        Returns:
            Comparison data dictionary
        """
        logger.info(f"Comparing protocols: {', '.join(protocol_slugs)}")
        
        # Get all protocols first to filter and avoid multiple API calls
        all_protocols = await self.get_protocols()
        
        # Filter to the requested protocols
        protocol_data = []
        for protocol in all_protocols:
            if protocol.get('slug') in protocol_slugs:
                filtered_protocol = {
                    'name': protocol.get('name'),
                    'slug': protocol.get('slug'),
                    'category': protocol.get('category'),
                }
                
                # Add requested metrics
                for metric in metrics:
                    if metric == 'tvl':
                        filtered_protocol['tvl'] = protocol.get('tvl', 0)
                    elif metric == 'mcap':
                        filtered_protocol['mcap'] = protocol.get('mcap', 0)
                    elif metric == 'fdv':
                        filtered_protocol['fdv'] = protocol.get('fdv', 0)
                
                # Calculate ratios if both metrics are available
                if 'tvl' in filtered_protocol and 'mcap' in filtered_protocol and filtered_protocol['tvl'] > 0:
                    filtered_protocol['mcap_tvl_ratio'] = filtered_protocol['mcap'] / filtered_protocol['tvl']
                
                if 'tvl' in filtered_protocol and 'fdv' in filtered_protocol and filtered_protocol['tvl'] > 0:
                    filtered_protocol['fdv_tvl_ratio'] = filtered_protocol['fdv'] / filtered_protocol['tvl']
                
                protocol_data.append(filtered_protocol)
        
        return {
            'protocols': protocol_data,
            'metrics': metrics,
            'timestamp': int(time.time())
        }
    
    async def find_top_yield_opportunities(
        self,
        min_tvl: float = 1000000,  # $1M minimum TVL
        min_apy: float = 5,  # 5% minimum APY
        max_apy: float = 100,  # 100% maximum APY (filter out likely unsustainable yields)
        chains: Optional[List[str]] = None,
        stablecoin_only: bool = False
    ) -> Dict[str, Any]:
        """
        Find top yield opportunities based on criteria.
        
        Args:
            min_tvl: Minimum TVL in USD
            min_apy: Minimum APY (as percentage)
            max_apy: Maximum APY (as percentage)
            chains: List of chains to include (None for all)
            stablecoin_only: Whether to only include stablecoin pools
        
        Returns:
            Top yield opportunities data
        """
        logger.info("Finding top yield opportunities")
        
        # Get all pools
        pools = await self.get_pools(tvl_min=min_tvl, apy_min=min_apy, apy_max=max_apy)
        
        # Filter by chain if specified
        if chains:
            pools = [pool for pool in pools if pool.get('chain') in chains]
        
        # Filter for stablecoin pools if requested
        if stablecoin_only:
            stablecoins = await self.get_stablecoins()
            stablecoin_symbols = [coin.get('symbol', '').upper() for coin in stablecoins.get('peggedAssets', [])]
            
            # Filter pools that contain stablecoins in their name
            stablecoin_pools = []
            for pool in pools:
                pool_symbol = pool.get('symbol', '').upper()
                if any(stable in pool_symbol for stable in stablecoin_symbols):
                    stablecoin_pools.append(pool)
            
            pools = stablecoin_pools
        
        # Sort by APY
        pools.sort(key=lambda x: x.get('apy', 0), reverse=True)
        
        return {
            'opportunities': pools[:20],  # Return top 20
            'criteria': {
                'min_tvl': min_tvl,
                'min_apy': min_apy,
                'max_apy': max_apy,
                'chains': chains,
                'stablecoin_only': stablecoin_only
            },
            'timestamp': int(time.time())
        }
    
    async def analyze_chain_dominance(self) -> Dict[str, Any]:
        """
        Analyze TVL dominance across different chains.
        
        Returns:
            Chain dominance analysis data
        """
        logger.info("Analyzing chain dominance")
        
        chains_data = await self.get_chains()
        
        # Calculate total TVL across all chains
        total_tvl = sum(chain.get('tvl', 0) for chain in chains_data)
        
        # Calculate dominance for each chain
        chain_dominance = []
        for chain in chains_data:
            tvl = chain.get('tvl', 0)
            dominance = (tvl / total_tvl * 100) if total_tvl > 0 else 0
            
            chain_dominance.append({
                'name': chain.get('name'),
                'tvl': tvl,
                'dominance': dominance
            })
        
        # Sort by dominance
        chain_dominance.sort(key=lambda x: x['dominance'], reverse=True)
        
        return {
            'total_tvl': total_tvl,
            'chains': chain_dominance,
            'timestamp': int(time.time())
        }
    
    async def track_protocol_growth(
        self,
        protocol_slug: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Track growth metrics for a protocol over time.
        
        Args:
            protocol_slug: Protocol slug/name
            days: Number of days to look back
        
        Returns:
            Protocol growth metrics
        """
        logger.info(f"Tracking growth for protocol: {protocol_slug} over {days} days")
        
        # Get current protocol data
        protocol_data = await self.get_protocol_data(protocol_slug)
        
        # Get historical TVL data
        to_timestamp = int(time.time())
        from_timestamp = to_timestamp - (days * 86400)  # days in seconds
        
        tvl_history = await self.get_protocol_tvl_history(
            protocol_slug=protocol_slug,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp
        )
        
        # Calculate growth metrics
        if not tvl_history:
            return {
                'protocol': protocol_slug,
                'error': 'No historical data available'
            }
        
        # Get first and last TVL data points
        first_tvl = tvl_history[0]['totalLiquidityUSD'] if tvl_history else 0
        last_tvl = tvl_history[-1]['totalLiquidityUSD'] if tvl_history else 0
        
        # Calculate absolute and percentage growth
        absolute_growth = last_tvl - first_tvl
        percentage_growth = ((last_tvl / first_tvl) - 1) * 100 if first_tvl > 0 else 0
        
        # Calculate volatility (standard deviation of daily changes)
        daily_changes = []
        for i in range(1, len(tvl_history)):
            prev_tvl = tvl_history[i-1]['totalLiquidityUSD']
            curr_tvl = tvl_history[i]['totalLiquidityUSD']
            if prev_tvl > 0:
                daily_change = ((curr_tvl / prev_tvl) - 1) * 100
                daily_changes.append(daily_change)
        
        volatility = 0
        if daily_changes:
            mean_change = sum(daily_changes) / len(daily_changes)
            sum_squared_diff = sum((x - mean_change) ** 2 for x in daily_changes)
            volatility = (sum_squared_diff / len(daily_changes)) ** 0.5
        
        return {
            'protocol': protocol_slug,
            'name': protocol_data.get('name', protocol_slug),
            'category': protocol_data.get('category', 'Unknown'),
            'current_tvl': last_tvl,
            'tvl_start': first_tvl,
            'absolute_growth': absolute_growth,
            'percentage_growth': percentage_growth,
            'volatility': volatility,
            'period_days': days,
            'tvl_history': tvl_history
        }
    
    # Neo4j Integration Methods
    
    async def store_protocol_in_neo4j(
        self,
        protocol_data: Dict[str, Any],
        store_tvl_history: bool = False
    ) -> Dict[str, Any]:
        """
        Store protocol data in Neo4j for further analysis.
        
        Args:
            protocol_data: Protocol data from get_protocol_data()
            store_tvl_history: Whether to store TVL history data
        
        Returns:
            Summary of stored data
        
        Raises:
            ValueError: If Neo4j client is not provided
        """
        if not self.neo4j_client:
            raise ValueError("Neo4j client is required to store results")
        
        if not self.neo4j_client.is_connected:
            await self.neo4j_client.connect()
        
        protocol_slug = protocol_data.get('slug', 'unknown')
        logger.info(f"Storing protocol data in Neo4j: {protocol_slug}")
        
        # Create protocol node
        protocol_props = {
            'slug': protocol_slug,
            'name': protocol_data.get('name', protocol_slug),
            'description': protocol_data.get('description', ''),
            'category': protocol_data.get('category', 'Unknown'),
            'tvl': protocol_data.get('tvl', 0),
            'mcap': protocol_data.get('mcap', 0),
            'fdv': protocol_data.get('fdv', 0),
            'updated_at': datetime.now().isoformat()
        }
        
        protocol_node = await self.neo4j_client.create_node(
            labels=['Protocol', 'DeFi'],
            properties=protocol_props
        )
        
        # Store chain relationships
        chain_relationships = 0
        for chain, chain_data in protocol_data.get('chainTvls', {}).items():
            if chain == 'tvl':  # Skip the total TVL entry
                continue
                
            # Create chain node if it doesn't exist
            chain_props = {
                'name': chain,
                'tvl': chain_data.get('tvl', 0),
                'updated_at': datetime.now().isoformat()
            }
            
            chain_node = await self.neo4j_client.create_node(
                labels=['Chain', 'Blockchain'],
                properties=chain_props
            )
            
            # Create relationship between protocol and chain
            await self.neo4j_client.create_relationship(
                from_node_id=protocol_node['id'],
                to_node_id=chain_node['id'],
                relationship_type='DEPLOYED_ON',
                properties={
                    'tvl': chain_data.get('tvl', 0),
                    'updated_at': datetime.now().isoformat()
                }
            )
            
            chain_relationships += 1
        
        # Store TVL history if requested
        tvl_history_points = 0
        if store_tvl_history:
            tvl_history = await self.get_protocol_tvl_history(protocol_slug)
            
            for point in tvl_history:
                tvl_props = {
                    'protocol_slug': protocol_slug,
                    'date': datetime.fromtimestamp(point.get('date', 0)).isoformat(),
                    'tvl': point.get('totalLiquidityUSD', 0),
                    'timestamp': point.get('date', 0)
                }
                
                await self.neo4j_client.create_node(
                    labels=['TVLDataPoint'],
                    properties=tvl_props
                )
                
                tvl_history_points += 1
        
        return {
            'protocol': protocol_slug,
            'protocol_node_id': protocol_node['id'],
            'chain_relationships': chain_relationships,
            'tvl_history_points': tvl_history_points
        }
    
    # Data Formatting Methods
    
    async def results_to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        Convert DefiLlama data to a pandas DataFrame.
        
        Args:
            data: Data from any DefiLlama API endpoint
        
        Returns:
            Pandas DataFrame with the results
        """
        # Handle different data structures based on endpoint
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try to extract a list from the dictionary
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    return pd.DataFrame(value)
            
            # If no list found, convert the dict itself to a DataFrame
            return pd.DataFrame([data])
        else:
            return pd.DataFrame()
    
    # Tool interface methods for CrewAI
    
    async def _run(
        self,
        action: str,
        **kwargs
    ) -> str:
        """
        Run the tool with the given action and parameters.
        This method is used by CrewAI to execute the tool.
        
        Args:
            action: Action to perform (e.g., 'get_protocol_tvl', 'find_yield_opportunities')
            **kwargs: Parameters for the action
        
        Returns:
            String representation of the results
        """
        try:
            # Map action to method
            if action == 'get_protocol_tvl':
                protocol = kwargs.get('protocol')
                if not protocol:
                    return "Error: Protocol slug is required"
                
                data = await self.get_protocol_data(protocol)
                tvl = data.get('tvl', 0)
                return f"Protocol {protocol} TVL: ${tvl:,.2f}"
            
            elif action == 'find_yield_opportunities':
                min_tvl = float(kwargs.get('min_tvl', 1000000))
                min_apy = float(kwargs.get('min_apy', 5))
                max_apy = float(kwargs.get('max_apy', 100))
                chains = kwargs.get('chains', '').split(',') if kwargs.get('chains') else None
                stablecoin_only = kwargs.get('stablecoin_only', 'false').lower() == 'true'
                
                data = await self.find_top_yield_opportunities(
                    min_tvl=min_tvl,
                    min_apy=min_apy,
                    max_apy=max_apy,
                    chains=chains,
                    stablecoin_only=stablecoin_only
                )
                
                opportunities = data.get('opportunities', [])
                if not opportunities:
                    return "No yield opportunities found matching the criteria"
                
                # Format results for agent consumption
                result_str = f"Top yield opportunities (min TVL: ${min_tvl:,.0f}, APY range: {min_apy:.1f}%-{max_apy:.1f}%):\n\n"
                result_str += f"{'Pool':<40} | {'Chain':<10} | {'APY (%)':<10} | {'TVL ($)':<15}\n"
                result_str += "-" * 80 + "\n"
                
                for pool in opportunities[:10]:  # Top 10 for readability
                    name = pool.get('symbol', 'Unknown')
                    chain = pool.get('chain', 'Unknown')
                    apy = pool.get('apy', 0)
                    tvl = pool.get('tvlUsd', 0)
                    
                    result_str += f"{name[:40]:<40} | {chain[:10]:<10} | {apy:<10.2f} | ${tvl:<15,.0f}\n"
                
                if len(opportunities) > 10:
                    result_str += f"\n... and {len(opportunities) - 10} more opportunities"
                
                return result_str
            
            elif action == 'analyze_protocol_growth':
                protocol = kwargs.get('protocol')
                days = int(kwargs.get('days', 30))
                
                if not protocol:
                    return "Error: Protocol slug is required"
                
                data = await self.track_protocol_growth(protocol, days)
                
                if 'error' in data:
                    return f"Error analyzing {protocol}: {data['error']}"
                
                # Format results for agent consumption
                result_str = f"Growth analysis for {data['name']} ({data['category']}) over {days} days:\n\n"
                result_str += f"Current TVL: ${data['current_tvl']:,.2f}\n"
                result_str += f"Starting TVL: ${data['tvl_start']:,.2f}\n"
                result_str += f"Absolute growth: ${data['absolute_growth']:,.2f}\n"
                result_str += f"Percentage growth: {data['percentage_growth']:.2f}%\n"
                result_str += f"TVL volatility: {data['volatility']:.2f}%\n"
                
                return result_str
            
            elif action == 'compare_protocols':
                protocols = kwargs.get('protocols', '').split(',')
                metrics = kwargs.get('metrics', 'tvl,mcap,fdv').split(',')
                
                if not protocols:
                    return "Error: At least one protocol slug is required"
                
                data = await self.compare_protocols(protocols, metrics)
                
                # Format results for agent consumption
                result_str = f"Protocol comparison ({', '.join(metrics)}):\n\n"
                
                # Create header
                header = "Protocol".ljust(20)
                for metric in metrics:
                    header += f" | {metric.upper()}".ljust(15)
                if 'tvl' in metrics and 'mcap' in metrics:
                    header += " | MCAP/TVL".ljust(15)
                
                result_str += header + "\n"
                result_str += "-" * len(header) + "\n"
                
                # Add protocol data
                for protocol in data.get('protocols', []):
                    line = protocol.get('name', 'Unknown').ljust(20)
                    
                    for metric in metrics:
                        value = protocol.get(metric, 0)
                        line += f" | ${value:,.2f}".ljust(15)
                    
                    if 'tvl' in metrics and 'mcap' in metrics and protocol.get('tvl', 0) > 0:
                        ratio = protocol.get('mcap_tvl_ratio', 0)
                        line += f" | {ratio:.2f}".ljust(15)
                    
                    result_str += line + "\n"
                
                return result_str
            
            else:
                return f"Unknown action: {action}. Available actions: get_protocol_tvl, find_yield_opportunities, analyze_protocol_growth, compare_protocols"
            
        except Exception as e:
            logger.error(f"Error executing DefiLlama action: {str(e)}")
            return f"Error executing {action}: {str(e)}"
