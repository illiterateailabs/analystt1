"""
Etherscan API integration for blockchain data analysis.

This module provides a tool for interacting with Etherscan APIs and similar
block explorer APIs (BSCScan, PolygonScan, etc.), allowing agents to retrieve
blockchain data, analyze transactions, and store results in Neo4j.

Documentation: https://docs.etherscan.io/
"""

import logging
import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlencode
import re

import aiohttp
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.config import settings
from backend.integrations.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

class EtherscanAPIConfig:
    """Configuration for Etherscan API and similar block explorers."""
    
    # Base URLs for different networks
    NETWORK_URLS = {
        'ethereum': 'https://api.etherscan.io/api',
        'goerli': 'https://api-goerli.etherscan.io/api',
        'sepolia': 'https://api-sepolia.etherscan.io/api',
        'bsc': 'https://api.bscscan.com/api',
        'polygon': 'https://api.polygonscan.com/api',
        'arbitrum': 'https://api.arbiscan.io/api',
        'optimism': 'https://api-optimistic.etherscan.io/api',
        'fantom': 'https://api.ftmscan.com/api',
        'avalanche': 'https://api.snowtrace.io/api',
    }
    
    # API Key from settings or environment
    API_KEYS = {
        'ethereum': getattr(settings, "etherscan_api_key", None),
        'goerli': getattr(settings, "etherscan_api_key", None),
        'sepolia': getattr(settings, "etherscan_api_key", None),
        'bsc': getattr(settings, "bscscan_api_key", None),
        'polygon': getattr(settings, "polygonscan_api_key", None),
        'arbitrum': getattr(settings, "arbiscan_api_key", None),
        'optimism': getattr(settings, "optimism_api_key", None),
        'fantom': getattr(settings, "ftmscan_api_key", None),
        'avalanche': getattr(settings, "snowtrace_api_key", None),
    }
    
    # Default to Ethereum API key if network-specific keys not provided
    for network in NETWORK_URLS:
        if not API_KEYS.get(network):
            API_KEYS[network] = getattr(settings, "etherscan_api_key", None)
    
    # Rate limiting settings
    MAX_REQUESTS_PER_SECOND = 5  # Free tier limit
    REQUEST_TIMEOUT = 20  # seconds
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_MIN_WAIT = 2  # seconds
    RETRY_MAX_WAIT = 10  # seconds

class EtherscanAPIError(Exception):
    """Exception for Etherscan API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

class EtherscanTool:
    """Tool for interacting with Etherscan API for blockchain data analysis."""
    
    def __init__(self, api_key: Optional[str] = None, neo4j_client: Optional[Neo4jClient] = None):
        """
        Initialize the Etherscan tool.
        
        Args:
            api_key: Etherscan API key (optional, falls back to config)
            neo4j_client: Neo4j client for storing results (optional)
        """
        self.config = EtherscanAPIConfig()
        self.default_api_key = api_key or self.config.API_KEYS['ethereum']
        self.neo4j_client = neo4j_client
        self._sessions = {}
        self._last_request_time = 0
        
        # Tool metadata for CrewAI
        self.name = "etherscan_tool"
        self.description = "Analyze blockchain data using Etherscan and similar block explorers"
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self):
        """Connect to Etherscan API by creating aiohttp sessions for each network."""
        for network in self.config.NETWORK_URLS:
            if network not in self._sessions or self._sessions[network].closed:
                self._sessions[network] = aiohttp.ClientSession()
                logger.debug(f"Created new aiohttp session for {network} API")
    
    async def close(self):
        """Close all aiohttp sessions."""
        for network, session in self._sessions.items():
            if not session.closed:
                await session.close()
                logger.debug(f"Closed aiohttp session for {network} API")
        
        self._sessions = {}
    
    async def _enforce_rate_limit(self):
        """
        Enforce rate limiting based on the configured limits.
        
        This method ensures we don't exceed the rate limit by
        waiting if necessary between requests.
        """
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        # If we've made a request too recently, wait
        min_interval = 1.0 / self.config.MAX_REQUESTS_PER_SECOND
        if time_since_last_request < min_interval:
            wait_time = min_interval - time_since_last_request
            await asyncio.sleep(wait_time)
        
        self._last_request_time = time.time()
    
    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(EtherscanAPIConfig.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=EtherscanAPIConfig.RETRY_MIN_WAIT,
            max=EtherscanAPIConfig.RETRY_MAX_WAIT
        )
    )
    async def _make_request(
        self,
        network: str,
        module: str,
        action: str,
        **params
    ) -> Dict:
        """
        Make a request to the Etherscan API with rate limiting and error handling.
        
        Args:
            network: Blockchain network (ethereum, bsc, polygon, etc.)
            module: API module (account, contract, etc.)
            action: API action (txlist, balance, etc.)
            **params: Additional parameters for the request
        
        Returns:
            Response data as dictionary
        
        Raises:
            EtherscanAPIError: If the request fails
        """
        await self.connect()  # Ensure sessions exist
        await self._enforce_rate_limit()  # Apply rate limiting
        
        # Get the appropriate session and API key for the network
        if network not in self._sessions:
            raise EtherscanAPIError(f"No session for network: {network}")
        
        session = self._sessions[network]
        api_key = self.config.API_KEYS.get(network, self.default_api_key)
        
        if not api_key:
            raise EtherscanAPIError(f"No API key for network: {network}")
        
        # Build the request URL
        base_url = self.config.NETWORK_URLS.get(network)
        if not base_url:
            raise EtherscanAPIError(f"Unsupported network: {network}")
        
        # Prepare query parameters
        query_params = {
            'module': module,
            'action': action,
            'apikey': api_key,
            **params
        }
        
        url = f"{base_url}?{urlencode(query_params)}"
        
        try:
            async with session.get(
                url=url,
                timeout=self.config.REQUEST_TIMEOUT
            ) as response:
                response_text = await response.text()
                
                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError:
                    raise EtherscanAPIError(
                        f"Invalid JSON response: {response_text[:100]}...",
                        status_code=response.status
                    )
                
                # Check for API errors
                if data.get('status') == '0':
                    error_message = data.get('message', 'Unknown error')
                    result = data.get('result', '')
                    
                    # Handle rate limit errors
                    if 'rate limit' in error_message.lower() or 'rate limit' in result.lower():
                        logger.warning(f"Rate limit hit for {network} API. Waiting before retry.")
                        await asyncio.sleep(1)  # Wait before retry
                        raise EtherscanAPIError(
                            f"Rate limit exceeded: {error_message}",
                            status_code=429,
                            response=data
                        )
                    
                    # Handle other API errors
                    raise EtherscanAPIError(
                        f"API error: {error_message}. Result: {result}",
                        response=data
                    )
                
                return data
        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during {network} API request: {str(e)}")
            raise EtherscanAPIError(f"HTTP error: {str(e)}")
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout during {network} API request to {module}/{action}")
            raise EtherscanAPIError("Request timed out")
    
    # Account Module Methods
    
    async def get_account_balance(
        self,
        address: str,
        network: str = 'ethereum',
        block: str = 'latest'
    ) -> Dict[str, Any]:
        """
        Get the Ether balance for an address.
        
        Args:
            address: Ethereum address
            network: Blockchain network
            block: Block number or 'latest'
        
        Returns:
            Balance information
        """
        logger.info(f"Getting {network} balance for address: {address}")
        
        data = await self._make_request(
            network=network,
            module='account',
            action='balance',
            address=address,
            tag=block
        )
        
        balance_wei = int(data.get('result', '0'))
        balance_eth = balance_wei / 1e18
        
        return {
            'address': address,
            'network': network,
            'balance_wei': balance_wei,
            'balance_eth': balance_eth,
            'block': block
        }
    
    async def get_token_balance(
        self,
        address: str,
        token_address: str,
        network: str = 'ethereum',
        block: str = 'latest'
    ) -> Dict[str, Any]:
        """
        Get the ERC-20 token balance for an address.
        
        Args:
            address: Ethereum address
            token_address: ERC-20 token contract address
            network: Blockchain network
            block: Block number or 'latest'
        
        Returns:
            Token balance information
        """
        logger.info(f"Getting {network} token balance for address: {address}, token: {token_address}")
        
        data = await self._make_request(
            network=network,
            module='account',
            action='tokenbalance',
            address=address,
            contractaddress=token_address,
            tag=block
        )
        
        # Get token details to determine decimals
        token_info = await self.get_token_info(token_address, network)
        decimals = int(token_info.get('decimals', 18))
        symbol = token_info.get('symbol', 'UNKNOWN')
        
        balance_raw = int(data.get('result', '0'))
        balance_token = balance_raw / (10 ** decimals)
        
        return {
            'address': address,
            'token_address': token_address,
            'token_symbol': symbol,
            'network': network,
            'balance_raw': balance_raw,
            'balance_token': balance_token,
            'decimals': decimals,
            'block': block
        }
    
    async def get_transactions(
        self,
        address: str,
        network: str = 'ethereum',
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100,
        sort: str = 'desc'
    ) -> Dict[str, Any]:
        """
        Get normal transactions for an address.
        
        Args:
            address: Ethereum address
            network: Blockchain network
            start_block: Starting block number
            end_block: Ending block number
            page: Page number
            offset: Number of results per page
            sort: Sort order ('asc' or 'desc')
        
        Returns:
            Transaction data
        """
        logger.info(f"Getting {network} transactions for address: {address}")
        
        data = await self._make_request(
            network=network,
            module='account',
            action='txlist',
            address=address,
            startblock=start_block,
            endblock=end_block,
            page=page,
            offset=offset,
            sort=sort
        )
        
        transactions = data.get('result', [])
        
        return {
            'address': address,
            'network': network,
            'transactions': transactions,
            'count': len(transactions),
            'page': page,
            'offset': offset
        }
    
    async def get_token_transfers(
        self,
        address: str,
        token_address: Optional[str] = None,
        network: str = 'ethereum',
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100,
        sort: str = 'desc'
    ) -> Dict[str, Any]:
        """
        Get ERC-20 token transfers for an address.
        
        Args:
            address: Ethereum address
            token_address: ERC-20 token contract address (optional)
            network: Blockchain network
            start_block: Starting block number
            end_block: Ending block number
            page: Page number
            offset: Number of results per page
            sort: Sort order ('asc' or 'desc')
        
        Returns:
            Token transfer data
        """
        logger.info(f"Getting {network} token transfers for address: {address}")
        
        params = {
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'page': page,
            'offset': offset,
            'sort': sort
        }
        
        if token_address:
            params['contractaddress'] = token_address
        
        data = await self._make_request(
            network=network,
            module='account',
            action='tokentx',
            **params
        )
        
        transfers = data.get('result', [])
        
        return {
            'address': address,
            'token_address': token_address,
            'network': network,
            'transfers': transfers,
            'count': len(transfers),
            'page': page,
            'offset': offset
        }
    
    async def get_internal_transactions(
        self,
        address: str,
        network: str = 'ethereum',
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100,
        sort: str = 'desc'
    ) -> Dict[str, Any]:
        """
        Get internal transactions for an address.
        
        Args:
            address: Ethereum address
            network: Blockchain network
            start_block: Starting block number
            end_block: Ending block number
            page: Page number
            offset: Number of results per page
            sort: Sort order ('asc' or 'desc')
        
        Returns:
            Internal transaction data
        """
        logger.info(f"Getting {network} internal transactions for address: {address}")
        
        data = await self._make_request(
            network=network,
            module='account',
            action='txlistinternal',
            address=address,
            startblock=start_block,
            endblock=end_block,
            page=page,
            offset=offset,
            sort=sort
        )
        
        transactions = data.get('result', [])
        
        return {
            'address': address,
            'network': network,
            'internal_transactions': transactions,
            'count': len(transactions),
            'page': page,
            'offset': offset
        }
    
    async def get_nft_transfers(
        self,
        address: str,
        network: str = 'ethereum',
        start_block: int = 0,
        end_block: int = 99999999,
        page: int = 1,
        offset: int = 100,
        sort: str = 'desc'
    ) -> Dict[str, Any]:
        """
        Get ERC-721 (NFT) transfers for an address.
        
        Args:
            address: Ethereum address
            network: Blockchain network
            start_block: Starting block number
            end_block: Ending block number
            page: Page number
            offset: Number of results per page
            sort: Sort order ('asc' or 'desc')
        
        Returns:
            NFT transfer data
        """
        logger.info(f"Getting {network} NFT transfers for address: {address}")
        
        data = await self._make_request(
            network=network,
            module='account',
            action='tokennfttx',
            address=address,
            startblock=start_block,
            endblock=end_block,
            page=page,
            offset=offset,
            sort=sort
        )
        
        transfers = data.get('result', [])
        
        return {
            'address': address,
            'network': network,
            'nft_transfers': transfers,
            'count': len(transfers),
            'page': page,
            'offset': offset
        }
    
    # Contract Module Methods
    
    async def get_contract_abi(
        self,
        address: str,
        network: str = 'ethereum'
    ) -> Dict[str, Any]:
        """
        Get the ABI for a verified smart contract.
        
        Args:
            address: Contract address
            network: Blockchain network
        
        Returns:
            Contract ABI data
        """
        logger.info(f"Getting {network} contract ABI for address: {address}")
        
        data = await self._make_request(
            network=network,
            module='contract',
            action='getabi',
            address=address
        )
        
        abi_string = data.get('result', '')
        
        # Parse ABI if it's a valid JSON
        abi = None
        if abi_string and abi_string != 'Contract source code not verified':
            try:
                abi = json.loads(abi_string)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse ABI for contract {address}")
        
        return {
            'address': address,
            'network': network,
            'abi': abi,
            'is_verified': abi is not None
        }
    
    async def get_contract_source_code(
        self,
        address: str,
        network: str = 'ethereum'
    ) -> Dict[str, Any]:
        """
        Get the source code for a verified smart contract.
        
        Args:
            address: Contract address
            network: Blockchain network
        
        Returns:
            Contract source code data
        """
        logger.info(f"Getting {network} contract source code for address: {address}")
        
        data = await self._make_request(
            network=network,
            module='contract',
            action='getsourcecode',
            address=address
        )
        
        source_info = data.get('result', [{}])[0]
        
        return {
            'address': address,
            'network': network,
            'name': source_info.get('ContractName', ''),
            'source_code': source_info.get('SourceCode', ''),
            'compiler_version': source_info.get('CompilerVersion', ''),
            'optimization_used': source_info.get('OptimizationUsed', ''),
            'runs': source_info.get('Runs', ''),
            'constructor_arguments': source_info.get('ConstructorArguments', ''),
            'is_verified': bool(source_info.get('SourceCode', ''))
        }
    
    # Transaction Module Methods
    
    async def get_transaction_receipt(
        self,
        tx_hash: str,
        network: str = 'ethereum'
    ) -> Dict[str, Any]:
        """
        Get the receipt for a transaction.
        
        Args:
            tx_hash: Transaction hash
            network: Blockchain network
        
        Returns:
            Transaction receipt data
        """
        logger.info(f"Getting {network} transaction receipt for hash: {tx_hash}")
        
        data = await self._make_request(
            network=network,
            module='proxy',
            action='eth_getTransactionReceipt',
            txhash=tx_hash
        )
        
        receipt = data.get('result', {})
        
        return {
            'tx_hash': tx_hash,
            'network': network,
            'receipt': receipt,
            'status': int(receipt.get('status', '0x0'), 16) if receipt.get('status') else None,
            'gas_used': int(receipt.get('gasUsed', '0x0'), 16) if receipt.get('gasUsed') else None,
            'block_number': int(receipt.get('blockNumber', '0x0'), 16) if receipt.get('blockNumber') else None,
            'logs': receipt.get('logs', [])
        }
    
    async def get_transaction_info(
        self,
        tx_hash: str,
        network: str = 'ethereum'
    ) -> Dict[str, Any]:
        """
        Get detailed information for a transaction.
        
        Args:
            tx_hash: Transaction hash
            network: Blockchain network
        
        Returns:
            Transaction information
        """
        logger.info(f"Getting {network} transaction info for hash: {tx_hash}")
        
        data = await self._make_request(
            network=network,
            module='proxy',
            action='eth_getTransactionByHash',
            txhash=tx_hash
        )
        
        tx_data = data.get('result', {})
        
        # Get receipt for additional information
        receipt_data = await self.get_transaction_receipt(tx_hash, network)
        
        # Combine transaction and receipt data
        return {
            'tx_hash': tx_hash,
            'network': network,
            'from': tx_data.get('from', ''),
            'to': tx_data.get('to', ''),
            'value': int(tx_data.get('value', '0x0'), 16) / 1e18 if tx_data.get('value') else 0,
            'value_wei': int(tx_data.get('value', '0x0'), 16) if tx_data.get('value') else 0,
            'gas': int(tx_data.get('gas', '0x0'), 16) if tx_data.get('gas') else 0,
            'gas_price': int(tx_data.get('gasPrice', '0x0'), 16) if tx_data.get('gasPrice') else 0,
            'nonce': int(tx_data.get('nonce', '0x0'), 16) if tx_data.get('nonce') else 0,
            'input': tx_data.get('input', ''),
            'block_number': int(tx_data.get('blockNumber', '0x0'), 16) if tx_data.get('blockNumber') else None,
            'status': receipt_data.get('status'),
            'gas_used': receipt_data.get('gas_used'),
            'logs': receipt_data.get('logs', [])
        }
    
    # Block Module Methods
    
    async def get_block_info(
        self,
        block_number: Union[int, str],
        network: str = 'ethereum'
    ) -> Dict[str, Any]:
        """
        Get information for a block.
        
        Args:
            block_number: Block number or 'latest'
            network: Blockchain network
        
        Returns:
            Block information
        """
        logger.info(f"Getting {network} block info for block: {block_number}")
        
        # Convert block number to hex if it's an integer
        if isinstance(block_number, int):
            block_param = hex(block_number)
        else:
            block_param = block_number
        
        data = await self._make_request(
            network=network,
            module='proxy',
            action='eth_getBlockByNumber',
            tag=block_param,
            boolean='true'  # Include full transaction objects
        )
        
        block_data = data.get('result', {})
        
        return {
            'block_number': int(block_data.get('number', '0x0'), 16) if block_data.get('number') else None,
            'network': network,
            'hash': block_data.get('hash', ''),
            'parent_hash': block_data.get('parentHash', ''),
            'timestamp': int(block_data.get('timestamp', '0x0'), 16) if block_data.get('timestamp') else None,
            'transactions': block_data.get('transactions', []),
            'transaction_count': len(block_data.get('transactions', [])),
            'gas_used': int(block_data.get('gasUsed', '0x0'), 16) if block_data.get('gasUsed') else None,
            'gas_limit': int(block_data.get('gasLimit', '0x0'), 16) if block_data.get('gasLimit') else None
        }
    
    # Token Module Methods
    
    async def get_token_info(
        self,
        token_address: str,
        network: str = 'ethereum'
    ) -> Dict[str, Any]:
        """
        Get information for an ERC-20 token.
        
        Args:
            token_address: Token contract address
            network: Blockchain network
        
        Returns:
            Token information
        """
        logger.info(f"Getting {network} token info for address: {token_address}")
        
        # Get token supply
        supply_data = await self._make_request(
            network=network,
            module='stats',
            action='tokensupply',
            contractaddress=token_address
        )
        
        # Get token name and symbol from contract ABI and source code
        contract_data = await self.get_contract_source_code(token_address, network)
        
        # Try to extract token details from source code if available
        name = ''
        symbol = ''
        decimals = 18  # Default for most tokens
        
        if contract_data.get('source_code'):
            # Extract token details from source code using regex
            name_match = re.search(r'name\s*=\s*[\'"]([^\'"]+)[\'"]', contract_data['source_code']) or \
                         re.search(r'name\s*\([^\)]*\)\s*[^{]*{[^}]*return\s*[\'"]([^\'"]+)[\'"]', contract_data['source_code'])
            
            symbol_match = re.search(r'symbol\s*=\s*[\'"]([^\'"]+)[\'"]', contract_data['source_code']) or \
                           re.search(r'symbol\s*\([^\)]*\)\s*[^{]*{[^}]*return\s*[\'"]([^\'"]+)[\'"]', contract_data['source_code'])
            
            decimals_match = re.search(r'decimals\s*=\s*(\d+)', contract_data['source_code']) or \
                             re.search(r'decimals\s*\([^\)]*\)\s*[^{]*{[^}]*return\s*(\d+)', contract_data['source_code'])
            
            if name_match:
                name = name_match.group(1)
            
            if symbol_match:
                symbol = symbol_match.group(1)
            
            if decimals_match:
                decimals = int(decimals_match.group(1))
        
        # If we couldn't extract from source, try to get from contract name
        if not name and contract_data.get('name'):
            name = contract_data['name']
        
        # Get token price if available (Ethereum only)
        price_usd = None
        if network == 'ethereum':
            try:
                price_data = await self._make_request(
                    network=network,
                    module='stats',
                    action='tokenpriceUSD',
                    contractaddress=token_address
                )
                price_usd = float(price_data.get('result', {}).get('ethusd', 0))
            except Exception as e:
                logger.warning(f"Failed to get token price for {token_address}: {e}")
        
        return {
            'address': token_address,
            'network': network,
            'name': name,
            'symbol': symbol,
            'decimals': decimals,
            'total_supply_raw': int(supply_data.get('result', '0')),
            'total_supply': int(supply_data.get('result', '0')) / (10 ** decimals),
            'price_usd': price_usd
        }
    
    # Gas Module Methods
    
    async def get_gas_price(
        self,
        network: str = 'ethereum'
    ) -> Dict[str, Any]:
        """
        Get current gas price.
        
        Args:
            network: Blockchain network
        
        Returns:
            Gas price information
        """
        logger.info(f"Getting {network} gas price")
        
        data = await self._make_request(
            network=network,
            module='proxy',
            action='eth_gasPrice'
        )
        
        gas_price_wei = int(data.get('result', '0x0'), 16)
        gas_price_gwei = gas_price_wei / 1e9
        
        return {
            'network': network,
            'gas_price_wei': gas_price_wei,
            'gas_price_gwei': gas_price_gwei,
            'timestamp': int(time.time())
        }
    
    async def get_gas_oracle(
        self,
        network: str = 'ethereum'
    ) -> Dict[str, Any]:
        """
        Get gas oracle information (safe, proposed, fast gas prices).
        
        Args:
            network: Blockchain network
        
        Returns:
            Gas oracle information
        """
        logger.info(f"Getting {network} gas oracle")
        
        data = await self._make_request(
            network=network,
            module='gastracker',
            action='gasoracle'
        )
        
        result = data.get('result', {})
        
        return {
            'network': network,
            'safe_gas_price': float(result.get('SafeGasPrice', 0)),
            'proposed_gas_price': float(result.get('ProposeGasPrice', 0)),
            'fast_gas_price': float(result.get('FastGasPrice', 0)),
            'last_block': int(result.get('LastBlock', 0)),
            'timestamp': int(time.time())
        }
    
    # Advanced Analysis Methods
    
    async def analyze_wallet(
        self,
        address: str,
        network: str = 'ethereum',
        include_tokens: bool = True,
        include_nfts: bool = False
    ) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of a wallet address.
        
        Args:
            address: Ethereum address
            network: Blockchain network
            include_tokens: Whether to include token balances
            include_nfts: Whether to include NFT holdings
        
        Returns:
            Wallet analysis data
        """
        logger.info(f"Analyzing {network} wallet: {address}")
        
        # Get ETH balance
        balance_data = await self.get_account_balance(address, network)
        
        # Get transaction history (last 100 transactions)
        tx_data = await self.get_transactions(address, network, offset=100)
        
        # Get token transfers (last 100 transfers)
        token_tx_data = await self.get_token_transfers(address, network, offset=100)
        
        # Initialize result
        result = {
            'address': address,
            'network': network,
            'balance_eth': balance_data['balance_eth'],
            'transaction_count': tx_data['count'],
            'token_transfer_count': token_tx_data['count'],
            'first_transaction': None,
            'last_transaction': None,
            'tokens': [],
            'nfts': []
        }
        
        # Find first and last transaction
        transactions = tx_data['transactions']
        if transactions:
            transactions.sort(key=lambda x: int(x.get('timeStamp', 0)))
            result['first_transaction'] = {
                'hash': transactions[0].get('hash', ''),
                'timestamp': int(transactions[0].get('timeStamp', 0)),
                'date': datetime.fromtimestamp(int(transactions[0].get('timeStamp', 0))).isoformat()
            }
            
            transactions.sort(key=lambda x: int(x.get('timeStamp', 0)), reverse=True)
            result['last_transaction'] = {
                'hash': transactions[0].get('hash', ''),
                'timestamp': int(transactions[0].get('timeStamp', 0)),
                'date': datetime.fromtimestamp(int(transactions[0].get('timeStamp', 0))).isoformat()
            }
        
        # Get token balances if requested
        if include_tokens:
            # Extract unique token addresses from transfers
            token_addresses = set()
            for transfer in token_tx_data['transfers']:
                token_addresses.add(transfer.get('contractAddress', ''))
            
            # Get balance for each token
            tokens = []
            for token_address in token_addresses:
                if not token_address:
                    continue
                
                try:
                    token_balance = await self.get_token_balance(address, token_address, network)
                    if token_balance['balance_token'] > 0:
                        tokens.append({
                            'address': token_address,
                            'symbol': token_balance['token_symbol'],
                            'balance': token_balance['balance_token'],
                            'decimals': token_balance['decimals']
                        })
                except Exception as e:
                    logger.warning(f"Failed to get token balance for {token_address}: {e}")
            
            result['tokens'] = tokens
        
        # Get NFT holdings if requested
        if include_nfts:
            try:
                nft_data = await self.get_nft_transfers(address, network, offset=100)
                
                # Extract unique NFT contracts and token IDs
                nfts = {}
                for transfer in nft_data['nft_transfers']:
                    contract = transfer.get('contractAddress', '')
                    token_id = transfer.get('tokenID', '')
                    token_name = transfer.get('tokenName', '')
                    
                    if contract and token_id:
                        key = f"{contract}_{token_id}"
                        
                        # Check if this is the current owner
                        is_owner = transfer.get('to', '').lower() == address.lower()
                        
                        if is_owner and key not in nfts:
                            nfts[key] = {
                                'contract_address': contract,
                                'token_id': token_id,
                                'name': token_name
                            }
                        elif not is_owner and key in nfts:
                            # No longer owns this NFT
                            del nfts[key]
                
                result['nfts'] = list(nfts.values())
            
            except Exception as e:
                logger.warning(f"Failed to get NFT holdings for {address}: {e}")
                result['nfts'] = []
        
        return result
    
    async def track_token_transfers(
        self,
        token_address: str,
        network: str = 'ethereum',
        hours: int = 24,
        min_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Track recent transfers for a token, optionally filtering by value.
        
        Args:
            token_address: Token contract address
            network: Blockchain network
            hours: Number of hours to look back
            min_value: Minimum transfer value in token units
        
        Returns:
            Token transfer tracking data
        """
        logger.info(f"Tracking {network} transfers for token: {token_address}")
        
        # Get token info
        token_info = await self.get_token_info(token_address, network)
        decimals = token_info.get('decimals', 18)
        
        # Calculate start block (approximate based on average block time)
        # Ethereum: ~13 seconds per block
        # BSC: ~3 seconds per block
        # Polygon: ~2 seconds per block
        blocks_per_hour = {
            'ethereum': 277,  # 3600 / 13
            'bsc': 1200,      # 3600 / 3
            'polygon': 1800,  # 3600 / 2
            'arbitrum': 277,  # Approximation
            'optimism': 277,  # Approximation
            'fantom': 1200,   # Approximation
            'avalanche': 1200 # Approximation
        }
        
        # Get current block
        current_block_data = await self._make_request(
            network=network,
            module='proxy',
            action='eth_blockNumber'
        )
        
        current_block = int(current_block_data.get('result', '0x0'), 16)
        
        # Calculate start block
        blocks_back = hours * blocks_per_hour.get(network, 277)
        start_block = max(0, current_block - blocks_back)
        
        # Get token transfers
        transfers_data = await self._make_request(
            network=network,
            module='account',
            action='tokentx',
            contractaddress=token_address,
            startblock=start_block,
            endblock=current_block,
            page=1,
            offset=500,  # Get more transfers to analyze
            sort='desc'
        )
        
        transfers = transfers_data.get('result', [])
        
        # Filter by value if requested
        if min_value is not None:
            min_value_raw = min_value * (10 ** decimals)
            transfers = [t for t in transfers if int(t.get('value', '0')) >= min_value_raw]
        
        # Process transfers
        processed_transfers = []
        for transfer in transfers:
            value_raw = int(transfer.get('value', '0'))
            value_token = value_raw / (10 ** decimals)
            
            processed_transfers.append({
                'hash': transfer.get('hash', ''),
                'from': transfer.get('from', ''),
                'to': transfer.get('to', ''),
                'value_raw': value_raw,
                'value_token': value_token,
                'block_number': int(transfer.get('blockNumber', '0')),
                'timestamp': int(transfer.get('timeStamp', '0')),
                'date': datetime.fromtimestamp(int(transfer.get('timeStamp', '0'))).isoformat()
            })
        
        # Calculate statistics
        total_volume = sum(t['value_token'] for t in processed_transfers)
        unique_senders = len(set(t['from'] for t in processed_transfers))
        unique_receivers = len(set(t['to'] for t in processed_transfers))
        
        return {
            'token_address': token_address,
            'token_name': token_info.get('name', ''),
            'token_symbol': token_info.get('symbol', ''),
            'network': network,
            'period_hours': hours,
            'transfer_count': len(processed_transfers),
            'total_volume': total_volume,
            'unique_senders': unique_senders,
            'unique_receivers': unique_receivers,
            'transfers': processed_transfers
        }
    
    async def analyze_contract_interactions(
        self,
        contract_address: str,
        network: str = 'ethereum',
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze interactions with a smart contract.
        
        Args:
            contract_address: Contract address
            network: Blockchain network
            days: Number of days to look back
        
        Returns:
            Contract interaction analysis
        """
        logger.info(f"Analyzing {network} interactions for contract: {contract_address}")
        
        # Get contract info
        contract_info = await self.get_contract_source_code(contract_address, network)
        
        # Calculate start block (approximate based on average block time)
        blocks_per_day = {
            'ethereum': 6648,    # 24 * 277
            'bsc': 28800,        # 24 * 1200
            'polygon': 43200,    # 24 * 1800
            'arbitrum': 6648,    # Approximation
            'optimism': 6648,    # Approximation
            'fantom': 28800,     # Approximation
            'avalanche': 28800   # Approximation
        }
        
        # Get current block
        current_block_data = await self._make_request(
            network=network,
            module='proxy',
            action='eth_blockNumber'
        )
        
        current_block = int(current_block_data.get('result', '0x0'), 16)
        
        # Calculate start block
        blocks_back = days * blocks_per_day.get(network, 6648)
        start_block = max(0, current_block - blocks_back)
        
        # Get transactions to the contract
        tx_data = await self.get_transactions(contract_address, network, start_block=start_block, offset=500)
        transactions = tx_data['transactions']
        
        # Process transactions
        function_calls = {}
        unique_callers = set()
        
        for tx in transactions:
            caller = tx.get('from', '')
            input_data = tx.get('input', '')
            
            unique_callers.add(caller)
            
            # Extract function signature (first 10 characters of input data, including '0x')
            if len(input_data) >= 10:
                function_sig = input_data[:10]
                function_calls[function_sig] = function_calls.get(function_sig, 0) + 1
        
        # Sort function calls by frequency
        sorted_functions = sorted(function_calls.items(), key=lambda x: x[1], reverse=True)
        
        # Try to decode function signatures if ABI is available
        decoded_functions = []
        if contract_info.get('abi'):
            for sig, count in sorted_functions:
                # Try to match the signature with functions in the ABI
                function_name = 'Unknown'
                for func in contract_info['abi']:
                    if func.get('type') == 'function':
                        # Calculate function signature
                        name = func.get('name', '')
                        inputs = func.get('inputs', [])
                        input_types = ','.join([inp.get('type', '') for inp in inputs])
                        full_sig = f"{name}({input_types})"
                        
                        # We'd need a proper keccak256 hash here, but this is a simplification
                        # In a real implementation, use web3.py or similar to calculate the signature
                        if name:
                            decoded_functions.append({
                                'signature': sig,
                                'name': name,
                                'count': count
                            })
                            break
                
                if function_name == 'Unknown':
                    decoded_functions.append({
                        'signature': sig,
                        'name': 'Unknown',
                        'count': count
                    })
        else:
            # If no ABI, just use raw signatures
            decoded_functions = [{'signature': sig, 'name': 'Unknown', 'count': count} for sig, count in sorted_functions]
        
        return {
            'contract_address': contract_address,
            'contract_name': contract_info.get('name', 'Unknown'),
            'network': network,
            'period_days': days,
            'transaction_count': len(transactions),
            'unique_callers': len(unique_callers),
            'is_verified': contract_info.get('is_verified', False),
            'function_calls': decoded_functions
        }
    
    async def find_related_addresses(
        self,
        address: str,
        network: str = 'ethereum',
        max_transactions: int = 200
    ) -> Dict[str, Any]:
        """
        Find addresses related to a given address through transactions.
        
        Args:
            address: Starting address
            network: Blockchain network
            max_transactions: Maximum number of transactions to analyze
        
        Returns:
            Related addresses data
        """
        logger.info(f"Finding addresses related to {address} on {network}")
        
        # Get transactions
        tx_data = await self.get_transactions(address, network, offset=max_transactions)
        transactions = tx_data['transactions']
        
        # Get token transfers
        token_tx_data = await self.get_token_transfers(address, network, offset=max_transactions)
        token_transfers = token_tx_data['transfers']
        
        # Process transactions and transfers to find related addresses
        related = {}
        
        # Process normal transactions
        for tx in transactions:
            from_addr = tx.get('from', '').lower()
            to_addr = tx.get('to', '').lower()
            value = int(tx.get('value', '0'))
            timestamp = int(tx.get('timeStamp', '0'))
            
            if from_addr == address.lower():
                # Outgoing transaction
                if to_addr and to_addr != address.lower():
                    if to_addr not in related:
                        related[to_addr] = {
                            'address': to_addr,
                            'outgoing_count': 0,
                            'outgoing_value': 0,
                            'incoming_count': 0,
                            'incoming_value': 0,
                            'first_interaction': timestamp,
                            'last_interaction': timestamp,
                            'token_transfers': 0
                        }
                    
                    related[to_addr]['outgoing_count'] += 1
                    related[to_addr]['outgoing_value'] += value
                    related[to_addr]['first_interaction'] = min(related[to_addr]['first_interaction'], timestamp)
                    related[to_addr]['last_interaction'] = max(related[to_addr]['last_interaction'], timestamp)
            
            elif to_addr == address.lower():
                # Incoming transaction
                if from_addr and from_addr != address.lower():
                    if from_addr not in related:
                        related[from_addr] = {
                            'address': from_addr,
                            'outgoing_count': 0,
                            'outgoing_value': 0,
                            'incoming_count': 0,
                            'incoming_value': 0,
                            'first_interaction': timestamp,
                            'last_interaction': timestamp,
                            'token_transfers': 0
                        }
                    
                    related[from_addr]['incoming_count'] += 1
                    related[from_addr]['incoming_value'] += value
                    related[from_addr]['first_interaction'] = min(related[from_addr]['first_interaction'], timestamp)
                    related[from_addr]['last_interaction'] = max(related[from_addr]['last_interaction'], timestamp)
        
        # Process token transfers
        for transfer in token_transfers:
            from_addr = transfer.get('from', '').lower()
            to_addr = transfer.get('to', '').lower()
            timestamp = int(transfer.get('timeStamp', '0'))
            
            if from_addr == address.lower():
                # Outgoing token transfer
                if to_addr and to_addr != address.lower():
                    if to_addr not in related:
                        related[to_addr] = {
                            'address': to_addr,
                            'outgoing_count': 0,
                            'outgoing_value': 0,
                            'incoming_count': 0,
                            'incoming_value': 0,
                            'first_interaction': timestamp,
                            'last_interaction': timestamp,
                            'token_transfers': 0
                        }
                    
                    related[to_addr]['token_transfers'] += 1
                    related[to_addr]['first_interaction'] = min(related[to_addr]['first_interaction'], timestamp)
                    related[to_addr]['last_interaction'] = max(related[to_addr]['last_interaction'], timestamp)
            
            elif to_addr == address.lower():
                # Incoming token transfer
                if from_addr and from_addr != address.lower():
                    if from_addr not in related:
                        related[from_addr] = {
                            'address': from_addr,
                            'outgoing_count': 0,
                            'outgoing_value': 0,
                            'incoming_count': 0,
                            'incoming_value': 0,
                            'first_interaction': timestamp,
                            'last_interaction': timestamp,
                            'token_transfers': 0
                        }
                    
                    related[from_addr]['token_transfers'] += 1
                    related[from_addr]['first_interaction'] = min(related[from_addr]['first_interaction'], timestamp)
                    related[from_addr]['last_interaction'] = max(related[from_addr]['last_interaction'], timestamp)
        
        # Convert to list and sort by interaction count
        related_list = list(related.values())
        related_list.sort(key=lambda x: (x['outgoing_count'] + x['incoming_count'] + x['token_transfers']), reverse=True)
        
        # Format timestamps to dates
        for r in related_list:
            r['first_interaction_date'] = datetime.fromtimestamp(r['first_interaction']).isoformat()
            r['last_interaction_date'] = datetime.fromtimestamp(r['last_interaction']).isoformat()
        
        return {
            'address': address,
            'network': network,
            'related_count': len(related_list),
            'related_addresses': related_list
        }
    
    # Neo4j Integration Methods
    
    async def store_wallet_in_neo4j(
        self,
        address: str,
        network: str = 'ethereum',
        include_transactions: bool = True,
        max_transactions: int = 100
    ) -> Dict[str, Any]:
        """
        Store wallet data in Neo4j for further analysis.
        
        Args:
            address: Ethereum address
            network: Blockchain network
            include_transactions: Whether to include transactions
            max_transactions: Maximum number of transactions to store
        
        Returns:
            Summary of stored data
        
        Raises:
            ValueError: If Neo4j client is not provided
        """
        if not self.neo4j_client:
            raise ValueError("Neo4j client is required to store results")
        
        if not self.neo4j_client.is_connected:
            await self.neo4j_client.connect()
        
        logger.info(f"Storing {network} wallet data in Neo4j: {address}")
        
        # Analyze wallet
        wallet_data = await self.analyze_wallet(address, network, include_tokens=True)
        
        # Create wallet node
        wallet_props = {
            'address': address,
            'network': network,
            'balance_eth': wallet_data['balance_eth'],
            'first_tx_date': wallet_data.get('first_transaction', {}).get('date'),
            'last_tx_date': wallet_data.get('last_transaction', {}).get('date'),
            'updated_at': datetime.now().isoformat()
        }
        
        wallet_node = await self.neo4j_client.create_node(
            labels=['Wallet', 'Blockchain'],
            properties=wallet_props
        )
        
        # Store token balances
        token_relationships = 0
        for token in wallet_data.get('tokens', []):
            # Create token node if it doesn't exist
            token_props = {
                'address': token['address'],
                'network': network,
                'symbol': token['symbol'],
                'decimals': token['decimals'],
                'updated_at': datetime.now().isoformat()
            }
            
            token_node = await self.neo4j_client.create_node(
                labels=['Token', 'ERC20'],
                properties=token_props
            )
            
            # Create relationship between wallet and token
            await self.neo4j_client.create_relationship(
                from_node_id=wallet_node['id'],
                to_node_id=token_node['id'],
                relationship_type='HOLDS',
                properties={
                    'balance': token['balance'],
                    'updated_at': datetime.now().isoformat()
                }
            )
            
            token_relationships += 1
        
        # Store transactions if requested
        transaction_count = 0
        if include_transactions:
            # Get transactions
            tx_data = await self.get_transactions(address, network, offset=max_transactions)
            transactions = tx_data['transactions']
            
            for tx in transactions:
                # Create transaction node
                tx_props = {
                    'hash': tx.get('hash', ''),
                    'network': network,
                    'from_address': tx.get('from', ''),
                    'to_address': tx.get('to', ''),
                    'value': int(tx.get('value', '0')) / 1e18,
                    'gas': int(tx.get('gas', '0')),
                    'gas_price': int(tx.get('gasPrice', '0')),
                    'block_number': int(tx.get('blockNumber', '0')),
                    'timestamp': int(tx.get('timeStamp', '0')),
                    'date': datetime.fromtimestamp(int(tx.get('timeStamp', '0'))).isoformat()
                }
                
                tx_node = await self.neo4j_client.create_node(
                    labels=['Transaction', 'Blockchain'],
                    properties=tx_props
                )
                
                # Create relationship between wallet and transaction
                if tx.get('from', '').lower() == address.lower():
                    await self.neo4j_client.create_relationship(
                        from_node_id=wallet_node['id'],
                        to_node_id=tx_node['id'],
                        relationship_type='SENT',
                        properties={
                            'timestamp': int(tx.get('timeStamp', '0')),
                            'date': datetime.fromtimestamp(int(tx.get('timeStamp', '0'))).isoformat()
                        }
                    )
                
                if tx.get('to', '').lower() == address.lower():
                    await self.neo4j_client.create_relationship(
                        from_node_id=tx_node['id'],
                        to_node_id=wallet_node['id'],
                        relationship_type='RECEIVED_BY',
                        properties={
                            'timestamp': int(tx.get('timeStamp', '0')),
                            'date': datetime.fromtimestamp(int(tx.get('timeStamp', '0'))).isoformat()
                        }
                    )
                
                transaction_count += 1
        
        return {
            'address': address,
            'network': network,
            'wallet_node_id': wallet_node['id'],
            'token_relationships': token_relationships,
            'transaction_count': transaction_count
        }
    
    # Data Formatting Methods
    
    async def results_to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        Convert Etherscan data to a pandas DataFrame.
        
        Args:
            data: Data from any Etherscan API endpoint
        
        Returns:
            Pandas DataFrame with the results
        """
        # Handle different data structures based on endpoint
        if isinstance(data, dict):
            # Try to extract a list from the dictionary
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    return pd.DataFrame(value)
            
            # If no list found, convert the dict itself to a DataFrame
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
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
            action: Action to perform (e.g., 'analyze_wallet', 'track_token_transfers')
            **kwargs: Parameters for the action
        
        Returns:
            String representation of the results
        """
        try:
            # Map action to method
            if action == 'analyze_wallet':
                address = kwargs.get('address')
                network = kwargs.get('network', 'ethereum')
                
                if not address:
                    return "Error: Wallet address is required"
                
                data = await self.analyze_wallet(
                    address=address,
                    network=network,
                    include_tokens=kwargs.get('include_tokens', True),
                    include_nfts=kwargs.get('include_nfts', False)
                )
                
                # Format results for agent consumption
                result_str = f"Wallet Analysis for {address} on {network}:\n\n"
                result_str += f"ETH Balance: {data['balance_eth']:.6f} ETH\n"
                
                if data.get('first_transaction'):
                    result_str += f"First Transaction: {data['first_transaction']['date']}\n"
                
                if data.get('last_transaction'):
                    result_str += f"Last Transaction: {data['last_transaction']['date']}\n"
                
                result_str += f"Transaction Count: {data['transaction_count']}\n"
                result_str += f"Token Transfer Count: {data['token_transfer_count']}\n\n"
                
                if data.get('tokens'):
                    result_str += f"Token Holdings ({len(data['tokens'])}):\n"
                    for token in data['tokens'][:10]:  # Show top 10 tokens
                        result_str += f"- {token['symbol']}: {token['balance']:.6f}\n"
                    
                    if len(data['tokens']) > 10:
                        result_str += f"... and {len(data['tokens']) - 10} more tokens\n"
                else:
                    result_str += "No token holdings found\n"
                
                if data.get('nfts'):
                    result_str += f"\nNFT Holdings ({len(data['nfts'])}):\n"
                    for nft in data['nfts'][:5]:  # Show top 5 NFTs
                        result_str += f"- {nft['name']} (ID: {nft['token_id']})\n"
                    
                    if len(data['nfts']) > 5:
                        result_str += f"... and {len(data['nfts']) - 5} more NFTs\n"
                
                return result_str
            
            elif action == 'track_token_transfers':
                token_address = kwargs.get('token_address')
                network = kwargs.get('network', 'ethereum')
                hours = int(kwargs.get('hours', 24))
                min_value = float(kwargs.get('min_value', 0)) if kwargs.get('min_value') else None
                
                if not token_address:
                    return "Error: Token address is required"
                
                data = await self.track_token_transfers(
                    token_address=token_address,
                    network=network,
                    hours=hours,
                    min_value=min_value
                )
                
                # Format results for agent consumption
                result_str = f"Token Transfer Analysis for {data['token_symbol']} ({data['token_name']}) on {network}:\n\n"
                result_str += f"Period: Last {hours} hours\n"
                result_str += f"Transfer Count: {data['transfer_count']}\n"
                result_str += f"Total Volume: {data['total_volume']:.2f} {data['token_symbol']}\n"
                result_str += f"Unique Senders: {data['unique_senders']}\n"
                result_str += f"Unique Receivers: {data['unique_receivers']}\n\n"
                
                if data.get('transfers'):
                    result_str += f"Recent Transfers ({min(5, len(data['transfers']))}):\n"
                    for tx in data['transfers'][:5]:  # Show top 5 transfers
                        result_str += f"- {tx['date']}: {tx['from'][:8]}...{tx['from'][-6:]} → {tx['to'][:8]}...{tx['to'][-6:]} ({tx['value_token']:.4f} {data['token_symbol']})\n"
                
                return result_str
            
            elif action == 'find_related_addresses':
                address = kwargs.get('address')
                network = kwargs.get('network', 'ethereum')
                max_transactions = int(kwargs.get('max_transactions', 200))
                
                if not address:
                    return "Error: Address is required"
                
                data = await self.find_related_addresses(
                    address=address,
                    network=network,
                    max_transactions=max_transactions
                )
                
                # Format results for agent consumption
                result_str = f"Related Addresses for {address} on {network}:\n\n"
                result_str += f"Found {data['related_count']} related addresses\n\n"
                
                if data.get('related_addresses'):
                    result_str += f"Top Related Addresses ({min(10, len(data['related_addresses']))}):\n"
                    for addr in data['related_addresses'][:10]:  # Show top 10 addresses
                        total_tx = addr['outgoing_count'] + addr['incoming_count'] + addr['token_transfers']
                        result_str += f"- {addr['address'][:8]}...{addr['address'][-6:]}: {total_tx} interactions "
                        result_str += f"(First: {addr['first_interaction_date']}, Last: {addr['last_interaction_date']})\n"
                
                return result_str
            
            elif action == 'analyze_contract':
                contract_address = kwargs.get('contract_address')
                network = kwargs.get('network', 'ethereum')
                days = int(kwargs.get('days', 7))
                
                if not contract_address:
                    return "Error: Contract address is required"
                
                data = await self.analyze_contract_interactions(
                    contract_address=contract_address,
                    network=network,
                    days=days
                )
                
                # Format results for agent consumption
                result_str = f"Contract Analysis for {data['contract_name']} ({contract_address}) on {network}:\n\n"
                result_str += f"Period: Last {days} days\n"
                result_str += f"Transaction Count: {data['transaction_count']}\n"
                result_str += f"Unique Callers: {data['unique_callers']}\n"
                result_str += f"Verified: {'Yes' if data['is_verified'] else 'No'}\n\n"
                
                if data.get('function_calls'):
                    result_str += f"Top Function Calls ({min(10, len(data['function_calls']))}):\n"
                    for func in data['function_calls'][:10]:  # Show top 10 functions
                        result_str += f"- {func['name']} ({func['signature']}): {func['count']} calls\n"
                
                return result_str
            
            else:
                return f"Unknown action: {action}. Available actions: analyze_wallet, track_token_transfers, find_related_addresses, analyze_contract"
            
        except Exception as e:
            logger.error(f"Error executing Etherscan action: {str(e)}")
            return f"Error executing {action}: {str(e)}"
