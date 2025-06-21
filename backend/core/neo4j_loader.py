"""
Neo4j Graph Database Loader Helpers

This module provides comprehensive helper functions for loading data into Neo4j,
with specific functions for different data types (balances, activity, tokens, relationships).
It serves as a single schema touch-point for the application, ensuring consistent
data modeling and efficient graph operations.

Features:
- Batch processing for large datasets
- Proper error handling and transaction management
- Conflict resolution strategies (MERGE vs CREATE)
- Progress tracking and metrics integration
- Support for different blockchain networks
- Data validation and transformation
- Integration with event system for observability
"""

import asyncio
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

import neo4j
from neo4j import GraphDatabase, Record, Result, Transaction
from neo4j.exceptions import ClientError, DatabaseError, ServiceUnavailable, TransientError
from pydantic import BaseModel, Field, validator

from backend.core.events import GraphAddEvent, publish_event
from backend.core.metrics import DatabaseMetrics
from backend.providers import get_provider

# Configure module logger
logger = logging.getLogger(__name__)

# Type aliases for clarity
CypherParams = Dict[str, Any]
ProgressCallback = Callable[[int, int, float], None]


class ConflictStrategy(str, Enum):
    """Strategy for handling conflicts when ingesting data."""
    MERGE = "MERGE"  # Update existing nodes/relationships
    CREATE = "CREATE"  # Always create new nodes/relationships
    MERGE_ON_KEY = "MERGE_ON_KEY"  # Merge on specific properties
    CREATE_UNIQUE = "CREATE_UNIQUE"  # Create only if doesn't exist (MERGE + ON CREATE only)


class ChainType(str, Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    BASE = "base"
    SOLANA = "solana"
    BINANCE = "binance"
    UNKNOWN = "unknown"


class ValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


class SchemaError(Exception):
    """Exception raised for schema-related errors."""
    pass


class GraphStats(BaseModel):
    """Statistics about a graph operation."""
    nodes_created: int = 0
    nodes_merged: int = 0
    relationships_created: int = 0
    relationships_merged: int = 0
    properties_set: int = 0
    labels_added: int = 0
    query_time_ms: float = 0.0
    batch_count: int = 0
    total_records: int = 0


class NodeAddress(BaseModel):
    """Model for blockchain addresses."""
    address: str
    chain: ChainType
    label: Optional[str] = None
    
    @validator('address')
    def validate_address(cls, v: str, values: Dict[str, Any]) -> str:
        """Validate address format based on chain."""
        chain = values.get('chain', ChainType.UNKNOWN)
        
        if chain == ChainType.ETHEREUM:
            # Ethereum addresses are 42 chars (0x + 40 hex chars)
            if not (v.startswith('0x') and len(v) == 42):
                raise ValueError("Invalid Ethereum address format")
        elif chain == ChainType.BITCOIN:
            # Basic Bitcoin address validation
            if not (len(v) >= 26 and len(v) <= 35):
                raise ValueError("Invalid Bitcoin address length")
        
        return v


class Balance(BaseModel):
    """Model for wallet balances."""
    address: str
    chain: ChainType
    asset: str
    amount: float
    usd_value: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    block_height: Optional[int] = None


class Activity(BaseModel):
    """Model for blockchain activity/transactions."""
    tx_hash: str
    chain: ChainType
    from_address: str
    to_address: str
    asset: Optional[str] = None
    amount: Optional[float] = None
    usd_value: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    block_height: int
    tx_fee: Optional[float] = None
    tx_status: str = "success"
    tx_type: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class Token(BaseModel):
    """Model for token information."""
    address: str
    chain: ChainType
    name: str
    symbol: str
    decimals: int
    total_supply: Optional[float] = None
    market_cap: Optional[float] = None
    type: str = "ERC20"  # ERC20, ERC721, etc.
    logo_url: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None


class Relationship(BaseModel):
    """Model for entity relationships."""
    from_address: str
    to_address: str
    chain: ChainType
    relationship_type: str
    properties: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    source: str = "analysis"


class BatchConfig(BaseModel):
    """Configuration for batch processing."""
    batch_size: int = 1000
    parallel_batches: int = 1
    timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: int = 1


class Neo4jLoader:
    """
    Helper class for loading data into Neo4j with comprehensive features.
    
    This class provides methods for ingesting different types of blockchain data
    into a Neo4j graph database, with support for batching, error handling,
    conflict resolution, and integration with metrics and events.
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        provider_id: str = "neo4j",
        environment: str = "development",
        version: str = "1.8.0-beta",
    ):
        """
        Initialize the Neo4j loader.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            provider_id: Provider ID for configuration
            environment: Environment name for metrics
            version: Application version for metrics
        """
        # Load configuration from provider registry if not provided
        if not uri or not username or not password:
            provider_config = get_provider(provider_id)
            if not provider_config:
                raise ValueError(f"Provider not found: {provider_id}")
            
            uri = provider_config.get("connection_uri")
            if not uri:
                raise ValueError(f"Neo4j URI not configured for provider: {provider_id}")
            
            auth_config = provider_config.get("auth", {})
            if auth_config:
                import os
                username_env = auth_config.get("username_env_var")
                password_env = auth_config.get("password_env_var")
                
                if username_env and password_env:
                    username = os.environ.get(username_env)
                    password = os.environ.get(password_env)
                    
                    if not username or not password:
                        raise ValueError(
                            f"Neo4j credentials not found in environment variables: "
                            f"{username_env}, {password_env}"
                        )
            
            database = provider_config.get("database_name", database)
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.environment = environment
        self.version = version
        self.provider_id = provider_id
        
        # Verify connection
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            logger.info(f"Connected to Neo4j database: {self.database}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self) -> None:
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()
    
    def _execute_query(
        self,
        query: str,
        params: Optional[CypherParams] = None,
        database: Optional[str] = None,
    ) -> Result:
        """
        Execute a Cypher query with metrics tracking.
        
        Args:
            query: Cypher query
            params: Query parameters
            database: Database name (defaults to self.database)
            
        Returns:
            Neo4j Result object
            
        Raises:
            DatabaseError: For database errors
        """
        db_name = database or self.database
        
        # Track database operation with metrics
        @DatabaseMetrics.track_operation(
            database="neo4j",
            operation="query",
            environment=self.environment,
            version=self.version,
        )
        def run_query() -> Result:
            with self.driver.session(database=db_name) as session:
                return session.run(query, params or {})
        
        try:
            return run_query()
        except (ServiceUnavailable, ClientError, DatabaseError) as e:
            logger.error(f"Neo4j query error: {e}")
            logger.debug(f"Failed query: {query}")
            logger.debug(f"Query params: {params}")
            raise DatabaseError(f"Neo4j query error: {e}")
    
    def _execute_query_with_retry(
        self,
        query: str,
        params: Optional[CypherParams] = None,
        database: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_seconds: int = 1,
    ) -> Result:
        """
        Execute a Cypher query with retry logic.
        
        Args:
            query: Cypher query
            params: Query parameters
            database: Database name
            max_retries: Maximum number of retry attempts
            retry_delay_seconds: Delay between retries in seconds
            
        Returns:
            Neo4j Result object
            
        Raises:
            DatabaseError: When all retries fail
        """
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                return self._execute_query(query, params, database)
            except TransientError as e:
                # Retry on transient errors (e.g., deadlocks)
                attempt += 1
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"Transient Neo4j error, retrying ({attempt}/{max_retries}): {e}"
                    )
                    time.sleep(retry_delay_seconds * (2 ** (attempt - 1)))  # Exponential backoff
            except (ServiceUnavailable, ClientError) as e:
                # Retry on connection issues
                if "connection refused" in str(e).lower() or "connection reset" in str(e).lower():
                    attempt += 1
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Neo4j connection error, retrying ({attempt}/{max_retries}): {e}"
                        )
                        time.sleep(retry_delay_seconds * (2 ** (attempt - 1)))
                else:
                    # Don't retry on other client errors
                    raise
            except Exception as e:
                # Don't retry on other errors
                raise
        
        # All retries failed
        raise DatabaseError(f"Neo4j query failed after {max_retries} attempts: {last_error}")
    
    def _process_result_stats(self, result: Result) -> GraphStats:
        """
        Process result statistics from a Neo4j query.
        
        Args:
            result: Neo4j Result object
            
        Returns:
            GraphStats object with operation statistics
        """
        counters = result.consume().counters
        
        return GraphStats(
            nodes_created=counters.nodes_created,
            nodes_merged=getattr(counters, "nodes_merged", 0),
            relationships_created=counters.relationships_created,
            relationships_merged=getattr(counters, "relationships_merged", 0),
            properties_set=counters.properties_set,
            labels_added=counters.labels_added,
            query_time_ms=result.consume().result_available_after,
        )
    
    def _publish_graph_event(self, stats: GraphStats, source: str, chain: Optional[str] = None) -> None:
        """
        Publish a graph event with operation statistics.
        
        Args:
            stats: GraphStats object with operation statistics
            source: Source of the data
            chain: Blockchain network
        """
        # Extract node and relationship types
        node_types = set()
        relationship_types = set()
        
        # Publish event
        publish_event(
            event_type="data.graph_add",
            data={
                "node_count": stats.nodes_created + stats.nodes_merged,
                "relationship_count": stats.relationships_created + stats.relationships_merged,
                "node_types": list(node_types),
                "relationship_types": list(relationship_types),
                "query_time_ms": stats.query_time_ms,
                "source": source,
                "chain": chain,
            },
        )
    
    def _validate_data(self, data: List[BaseModel]) -> Tuple[List[BaseModel], List[Tuple[BaseModel, str]]]:
        """
        Validate a list of data models.
        
        Args:
            data: List of data models to validate
            
        Returns:
            Tuple of (valid_data, invalid_data_with_errors)
        """
        valid_data = []
        invalid_data = []
        
        for item in data:
            try:
                # Pydantic models validate on initialization, but we can force it again
                item_dict = item.dict()
                item.__class__(**item_dict)
                valid_data.append(item)
            except Exception as e:
                invalid_data.append((item, str(e)))
        
        return valid_data, invalid_data
    
    def _chunk_data(self, data: List[Any], batch_size: int) -> List[List[Any]]:
        """
        Split data into chunks for batch processing.
        
        Args:
            data: List of data to chunk
            batch_size: Size of each chunk
            
        Returns:
            List of data chunks
        """
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    async def _process_batches_parallel(
        self,
        batches: List[List[Any]],
        process_func: Callable[[List[Any]], GraphStats],
        parallel_batches: int,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GraphStats:
        """
        Process batches in parallel using asyncio.
        
        Args:
            batches: List of data batches
            process_func: Function to process each batch
            parallel_batches: Number of batches to process in parallel
            progress_callback: Callback for progress updates
            
        Returns:
            Combined GraphStats for all batches
        """
        semaphore = asyncio.Semaphore(parallel_batches)
        total_batches = len(batches)
        processed_batches = 0
        combined_stats = GraphStats()
        
        async def process_batch(batch: List[Any], batch_index: int) -> GraphStats:
            async with semaphore:
                # Run the processing function in a thread pool
                loop = asyncio.get_event_loop()
                stats = await loop.run_in_executor(None, process_func, batch)
                
                nonlocal processed_batches
                processed_batches += 1
                
                # Update progress
                if progress_callback:
                    progress_callback(
                        processed_batches,
                        total_batches,
                        processed_batches / total_batches * 100,
                    )
                
                return stats
        
        # Create tasks for all batches
        tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
        
        # Wait for all tasks to complete
        batch_stats = await asyncio.gather(*tasks)
        
        # Combine stats
        for stats in batch_stats:
            combined_stats.nodes_created += stats.nodes_created
            combined_stats.nodes_merged += stats.nodes_merged
            combined_stats.relationships_created += stats.relationships_created
            combined_stats.relationships_merged += stats.relationships_merged
            combined_stats.properties_set += stats.properties_set
            combined_stats.labels_added += stats.labels_added
            combined_stats.query_time_ms += stats.query_time_ms
        
        combined_stats.batch_count = total_batches
        combined_stats.total_records = sum(len(batch) for batch in batches)
        
        return combined_stats
    
    def ingest_balances(
        self,
        balances: List[Balance],
        conflict_strategy: ConflictStrategy = ConflictStrategy.MERGE,
        batch_config: Optional[BatchConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GraphStats:
        """
        Ingest wallet balance data into the graph database.
        
        Args:
            balances: List of Balance objects
            conflict_strategy: Strategy for handling conflicts
            batch_config: Configuration for batch processing
            progress_callback: Callback for progress updates
            
        Returns:
            GraphStats with operation statistics
            
        Raises:
            ValidationError: If data validation fails
            DatabaseError: For database errors
        """
        # Validate data
        valid_balances, invalid_balances = self._validate_data(balances)
        
        if invalid_balances:
            error_msg = f"{len(invalid_balances)} invalid balance records found"
            logger.warning(f"{error_msg}: {invalid_balances[:5]}")
            if len(invalid_balances) == len(balances):
                raise ValidationError(f"All balance records are invalid: {invalid_balances[0][1]}")
        
        if not valid_balances:
            logger.warning("No valid balance records to ingest")
            return GraphStats()
        
        # Use default batch config if not provided
        config = batch_config or BatchConfig()
        
        # Split data into batches
        batches = self._chunk_data(valid_balances, config.batch_size)
        
        # Define batch processing function
        def process_batch(batch: List[Balance]) -> GraphStats:
            # Build Cypher query based on conflict strategy
            if conflict_strategy == ConflictStrategy.MERGE:
                query = """
                UNWIND $balances AS balance
                MERGE (a:Address {address: balance.address, chain: balance.chain})
                MERGE (asset:Asset {symbol: balance.asset, chain: balance.chain})
                MERGE (a)-[r:HOLDS]->(asset)
                SET r.amount = balance.amount,
                    r.usd_value = balance.usd_value,
                    r.last_updated = balance.timestamp,
                    r.block_height = balance.block_height,
                    a.last_updated = balance.timestamp
                """
            elif conflict_strategy == ConflictStrategy.CREATE:
                query = """
                UNWIND $balances AS balance
                CREATE (a:Address {address: balance.address, chain: balance.chain})
                MERGE (asset:Asset {symbol: balance.asset, chain: balance.chain})
                CREATE (a)-[r:HOLDS]->(asset)
                SET r.amount = balance.amount,
                    r.usd_value = balance.usd_value,
                    r.last_updated = balance.timestamp,
                    r.block_height = balance.block_height,
                    a.last_updated = balance.timestamp
                """
            elif conflict_strategy == ConflictStrategy.MERGE_ON_KEY:
                query = """
                UNWIND $balances AS balance
                MERGE (a:Address {address: balance.address, chain: balance.chain})
                MERGE (asset:Asset {symbol: balance.asset, chain: balance.chain})
                MERGE (a)-[r:HOLDS {asset: balance.asset}]->(asset)
                SET r.amount = balance.amount,
                    r.usd_value = balance.usd_value,
                    r.last_updated = balance.timestamp,
                    r.block_height = balance.block_height,
                    a.last_updated = balance.timestamp
                """
            else:  # CREATE_UNIQUE
                query = """
                UNWIND $balances AS balance
                MERGE (a:Address {address: balance.address, chain: balance.chain})
                MERGE (asset:Asset {symbol: balance.asset, chain: balance.chain})
                MERGE (a)-[r:HOLDS {asset: balance.asset}]->(asset)
                ON CREATE SET r.amount = balance.amount,
                             r.usd_value = balance.usd_value,
                             r.last_updated = balance.timestamp,
                             r.block_height = balance.block_height,
                             a.last_updated = balance.timestamp
                """
            
            # Convert balances to dictionaries for Neo4j
            balance_dicts = [b.dict() for b in batch]
            
            # Execute query with retry
            start_time = time.time()
            result = self._execute_query_with_retry(
                query,
                {"balances": balance_dicts},
                max_retries=config.retry_attempts,
                retry_delay_seconds=config.retry_delay_seconds,
            )
            
            # Process result statistics
            stats = self._process_result_stats(result)
            
            # Track query time
            stats.query_time_ms = (time.time() - start_time) * 1000
            
            return stats
        
        # Process batches in parallel
        if config.parallel_batches > 1:
            loop = asyncio.get_event_loop()
            combined_stats = loop.run_until_complete(
                self._process_batches_parallel(
                    batches,
                    process_batch,
                    config.parallel_batches,
                    progress_callback,
                )
            )
        else:
            # Process batches sequentially
            combined_stats = GraphStats()
            total_batches = len(batches)
            
            for i, batch in enumerate(batches):
                stats = process_batch(batch)
                
                # Update combined stats
                combined_stats.nodes_created += stats.nodes_created
                combined_stats.nodes_merged += stats.nodes_merged
                combined_stats.relationships_created += stats.relationships_created
                combined_stats.relationships_merged += stats.relationships_merged
                combined_stats.properties_set += stats.properties_set
                combined_stats.labels_added += stats.labels_added
                combined_stats.query_time_ms += stats.query_time_ms
                
                # Update progress
                if progress_callback:
                    progress_callback(i + 1, total_batches, (i + 1) / total_batches * 100)
            
            combined_stats.batch_count = total_batches
            combined_stats.total_records = len(valid_balances)
        
        # Get a sample chain for the event
        sample_chain = valid_balances[0].chain.value if valid_balances else None
        
        # Publish graph event
        self._publish_graph_event(combined_stats, "balance_ingest", sample_chain)
        
        logger.info(
            f"Ingested {len(valid_balances)} balances "
            f"({combined_stats.nodes_created + combined_stats.nodes_merged} nodes, "
            f"{combined_stats.relationships_created + combined_stats.relationships_merged} relationships) "
            f"in {combined_stats.batch_count} batches"
        )
        
        return combined_stats
    
    def ingest_activity(
        self,
        activities: List[Activity],
        conflict_strategy: ConflictStrategy = ConflictStrategy.MERGE,
        batch_config: Optional[BatchConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GraphStats:
        """
        Ingest blockchain activity/transaction data into the graph database.
        
        Args:
            activities: List of Activity objects
            conflict_strategy: Strategy for handling conflicts
            batch_config: Configuration for batch processing
            progress_callback: Callback for progress updates
            
        Returns:
            GraphStats with operation statistics
            
        Raises:
            ValidationError: If data validation fails
            DatabaseError: For database errors
        """
        # Validate data
        valid_activities, invalid_activities = self._validate_data(activities)
        
        if invalid_activities:
            error_msg = f"{len(invalid_activities)} invalid activity records found"
            logger.warning(f"{error_msg}: {invalid_activities[:5]}")
            if len(invalid_activities) == len(activities):
                raise ValidationError(f"All activity records are invalid: {invalid_activities[0][1]}")
        
        if not valid_activities:
            logger.warning("No valid activity records to ingest")
            return GraphStats()
        
        # Use default batch config if not provided
        config = batch_config or BatchConfig()
        
        # Split data into batches
        batches = self._chunk_data(valid_activities, config.batch_size)
        
        # Define batch processing function
        def process_batch(batch: List[Activity]) -> GraphStats:
            # Build Cypher query based on conflict strategy
            if conflict_strategy == ConflictStrategy.MERGE:
                query = """
                UNWIND $activities AS activity
                MERGE (tx:Transaction {tx_hash: activity.tx_hash, chain: activity.chain})
                MERGE (from:Address {address: activity.from_address, chain: activity.chain})
                MERGE (to:Address {address: activity.to_address, chain: activity.chain})
                MERGE (from)-[sent:SENT]->(tx)
                MERGE (tx)-[received:RECEIVED]->(to)
                SET tx.timestamp = activity.timestamp,
                    tx.block_height = activity.block_height,
                    tx.status = activity.tx_status,
                    tx.tx_type = activity.tx_type,
                    tx.tx_fee = activity.tx_fee
                
                WITH tx, activity
                FOREACH (ignoreMe IN CASE WHEN activity.asset IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (asset:Asset {symbol: activity.asset, chain: activity.chain})
                    MERGE (tx)-[transfer:TRANSFERS]->(asset)
                    SET transfer.amount = activity.amount,
                        transfer.usd_value = activity.usd_value
                )
                
                WITH tx, activity
                FOREACH (ignoreMe IN CASE WHEN activity.data IS NOT NULL THEN [1] ELSE [] END |
                    SET tx += activity.data
                )
                """
            elif conflict_strategy == ConflictStrategy.CREATE:
                query = """
                UNWIND $activities AS activity
                CREATE (tx:Transaction {
                    tx_hash: activity.tx_hash,
                    chain: activity.chain,
                    timestamp: activity.timestamp,
                    block_height: activity.block_height,
                    status: activity.tx_status,
                    tx_type: activity.tx_type,
                    tx_fee: activity.tx_fee
                })
                
                MERGE (from:Address {address: activity.from_address, chain: activity.chain})
                MERGE (to:Address {address: activity.to_address, chain: activity.chain})
                
                CREATE (from)-[sent:SENT]->(tx)
                CREATE (tx)-[received:RECEIVED]->(to)
                
                WITH tx, activity
                FOREACH (ignoreMe IN CASE WHEN activity.asset IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (asset:Asset {symbol: activity.asset, chain: activity.chain})
                    CREATE (tx)-[transfer:TRANSFERS]->(asset)
                    SET transfer.amount = activity.amount,
                        transfer.usd_value = activity.usd_value
                )
                
                WITH tx, activity
                FOREACH (ignoreMe IN CASE WHEN activity.data IS NOT NULL THEN [1] ELSE [] END |
                    SET tx += activity.data
                )
                """
            else:  # MERGE_ON_KEY or CREATE_UNIQUE
                query = """
                UNWIND $activities AS activity
                MERGE (tx:Transaction {tx_hash: activity.tx_hash, chain: activity.chain})
                ON CREATE SET tx.timestamp = activity.timestamp,
                             tx.block_height = activity.block_height,
                             tx.status = activity.tx_status,
                             tx.tx_type = activity.tx_type,
                             tx.tx_fee = activity.tx_fee
                
                MERGE (from:Address {address: activity.from_address, chain: activity.chain})
                MERGE (to:Address {address: activity.to_address, chain: activity.chain})
                
                MERGE (from)-[sent:SENT]->(tx)
                MERGE (tx)-[received:RECEIVED]->(to)
                
                WITH tx, activity
                FOREACH (ignoreMe IN CASE WHEN activity.asset IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (asset:Asset {symbol: activity.asset, chain: activity.chain})
                    MERGE (tx)-[transfer:TRANSFERS {asset: activity.asset}]->(asset)
                    ON CREATE SET transfer.amount = activity.amount,
                                 transfer.usd_value = activity.usd_value
                )
                
                WITH tx, activity
                FOREACH (ignoreMe IN CASE WHEN activity.data IS NOT NULL THEN [1] ELSE [] END |
                    SET tx += activity.data
                )
                """
            
            # Convert activities to dictionaries for Neo4j
            activity_dicts = [a.dict() for a in batch]
            
            # Execute query with retry
            start_time = time.time()
            result = self._execute_query_with_retry(
                query,
                {"activities": activity_dicts},
                max_retries=config.retry_attempts,
                retry_delay_seconds=config.retry_delay_seconds,
            )
            
            # Process result statistics
            stats = self._process_result_stats(result)
            
            # Track query time
            stats.query_time_ms = (time.time() - start_time) * 1000
            
            return stats
        
        # Process batches in parallel
        if config.parallel_batches > 1:
            loop = asyncio.get_event_loop()
            combined_stats = loop.run_until_complete(
                self._process_batches_parallel(
                    batches,
                    process_batch,
                    config.parallel_batches,
                    progress_callback,
                )
            )
        else:
            # Process batches sequentially
            combined_stats = GraphStats()
            total_batches = len(batches)
            
            for i, batch in enumerate(batches):
                stats = process_batch(batch)
                
                # Update combined stats
                combined_stats.nodes_created += stats.nodes_created
                combined_stats.nodes_merged += stats.nodes_merged
                combined_stats.relationships_created += stats.relationships_created
                combined_stats.relationships_merged += stats.relationships_merged
                combined_stats.properties_set += stats.properties_set
                combined_stats.labels_added += stats.labels_added
                combined_stats.query_time_ms += stats.query_time_ms
                
                # Update progress
                if progress_callback:
                    progress_callback(i + 1, total_batches, (i + 1) / total_batches * 100)
            
            combined_stats.batch_count = total_batches
            combined_stats.total_records = len(valid_activities)
        
        # Get a sample chain for the event
        sample_chain = valid_activities[0].chain.value if valid_activities else None
        
        # Publish graph event
        self._publish_graph_event(combined_stats, "activity_ingest", sample_chain)
        
        logger.info(
            f"Ingested {len(valid_activities)} activities "
            f"({combined_stats.nodes_created + combined_stats.nodes_merged} nodes, "
            f"{combined_stats.relationships_created + combined_stats.relationships_merged} relationships) "
            f"in {combined_stats.batch_count} batches"
        )
        
        return combined_stats
    
    def ingest_tokens(
        self,
        tokens: List[Token],
        conflict_strategy: ConflictStrategy = ConflictStrategy.MERGE,
        batch_config: Optional[BatchConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GraphStats:
        """
        Ingest token information into the graph database.
        
        Args:
            tokens: List of Token objects
            conflict_strategy: Strategy for handling conflicts
            batch_config: Configuration for batch processing
            progress_callback: Callback for progress updates
            
        Returns:
            GraphStats with operation statistics
            
        Raises:
            ValidationError: If data validation fails
            DatabaseError: For database errors
        """
        # Validate data
        valid_tokens, invalid_tokens = self._validate_data(tokens)
        
        if invalid_tokens:
            error_msg = f"{len(invalid_tokens)} invalid token records found"
            logger.warning(f"{error_msg}: {invalid_tokens[:5]}")
            if len(invalid_tokens) == len(tokens):
                raise ValidationError(f"All token records are invalid: {invalid_tokens[0][1]}")
        
        if not valid_tokens:
            logger.warning("No valid token records to ingest")
            return GraphStats()
        
        # Use default batch config if not provided
        config = batch_config or BatchConfig()
        
        # Split data into batches
        batches = self._chunk_data(valid_tokens, config.batch_size)
        
        # Define batch processing function
        def process_batch(batch: List[Token]) -> GraphStats:
            # Build Cypher query based on conflict strategy
            if conflict_strategy == ConflictStrategy.MERGE:
                query = """
                UNWIND $tokens AS token
                MERGE (t:Token {address: token.address, chain: token.chain})
                SET t.name = token.name,
                    t.symbol = token.symbol,
                    t.decimals = token.decimals,
                    t.total_supply = token.total_supply,
                    t.market_cap = token.market_cap,
                    t.type = token.type,
                    t.logo_url = token.logo_url,
                    t.website = token.website,
                    t.description = token.description,
                    t.last_updated = datetime()
                
                MERGE (asset:Asset {symbol: token.symbol, chain: token.chain})
                MERGE (t)-[r:IMPLEMENTS]->(asset)
                """
            elif conflict_strategy == ConflictStrategy.CREATE:
                query = """
                UNWIND $tokens AS token
                CREATE (t:Token {
                    address: token.address,
                    chain: token.chain,
                    name: token.name,
                    symbol: token.symbol,
                    decimals: token.decimals,
                    total_supply: token.total_supply,
                    market_cap: token.market_cap,
                    type: token.type,
                    logo_url: token.logo_url,
                    website: token.website,
                    description: token.description,
                    last_updated: datetime()
                })
                
                MERGE (asset:Asset {symbol: token.symbol, chain: token.chain})
                CREATE (t)-[r:IMPLEMENTS]->(asset)
                """
            else:  # MERGE_ON_KEY or CREATE_UNIQUE
                query = """
                UNWIND $tokens AS token
                MERGE (t:Token {address: token.address, chain: token.chain})
                ON CREATE SET t.name = token.name,
                             t.symbol = token.symbol,
                             t.decimals = token.decimals,
                             t.total_supply = token.total_supply,
                             t.market_cap = token.market_cap,
                             t.type = token.type,
                             t.logo_url = token.logo_url,
                             t.website = token.website,
                             t.description = token.description,
                             t.last_updated = datetime()
                ON MATCH SET t.name = token.name,
                            t.decimals = token.decimals,
                            t.total_supply = token.total_supply,
                            t.market_cap = token.market_cap,
                            t.type = token.type,
                            t.logo_url = token.logo_url,
                            t.website = token.website,
                            t.description = token.description,
                            t.last_updated = datetime()
                
                MERGE (asset:Asset {symbol: token.symbol, chain: token.chain})
                MERGE (t)-[r:IMPLEMENTS]->(asset)
                """
            
            # Convert tokens to dictionaries for Neo4j
            token_dicts = [t.dict() for t in batch]
            
            # Execute query with retry
            start_time = time.time()
            result = self._execute_query_with_retry(
                query,
                {"tokens": token_dicts},
                max_retries=config.retry_attempts,
                retry_delay_seconds=config.retry_delay_seconds,
            )
            
            # Process result statistics
            stats = self._process_result_stats(result)
            
            # Track query time
            stats.query_time_ms = (time.time() - start_time) * 1000
            
            return stats
        
        # Process batches in parallel
        if config.parallel_batches > 1:
            loop = asyncio.get_event_loop()
            combined_stats = loop.run_until_complete(
                self._process_batches_parallel(
                    batches,
                    process_batch,
                    config.parallel_batches,
                    progress_callback,
                )
            )
        else:
            # Process batches sequentially
            combined_stats = GraphStats()
            total_batches = len(batches)
            
            for i, batch in enumerate(batches):
                stats = process_batch(batch)
                
                # Update combined stats
                combined_stats.nodes_created += stats.nodes_created
                combined_stats.nodes_merged += stats.nodes_merged
                combined_stats.relationships_created += stats.relationships_created
                combined_stats.relationships_merged += stats.relationships_merged
                combined_stats.properties_set += stats.properties_set
                combined_stats.labels_added += stats.labels_added
                combined_stats.query_time_ms += stats.query_time_ms
                
                # Update progress
                if progress_callback:
                    progress_callback(i + 1, total_batches, (i + 1) / total_batches * 100)
            
            combined_stats.batch_count = total_batches
            combined_stats.total_records = len(valid_tokens)
        
        # Get a sample chain for the event
        sample_chain = valid_tokens[0].chain.value if valid_tokens else None
        
        # Publish graph event
        self._publish_graph_event(combined_stats, "token_ingest", sample_chain)
        
        logger.info(
            f"Ingested {len(valid_tokens)} tokens "
            f"({combined_stats.nodes_created + combined_stats.nodes_merged} nodes, "
            f"{combined_stats.relationships_created + combined_stats.relationships_merged} relationships) "
            f"in {combined_stats.batch_count} batches"
        )
        
        return combined_stats
    
    def ingest_relationships(
        self,
        relationships: List[Relationship],
        conflict_strategy: ConflictStrategy = ConflictStrategy.MERGE,
        batch_config: Optional[BatchConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GraphStats:
        """
        Ingest entity relationships into the graph database.
        
        Args:
            relationships: List of Relationship objects
            conflict_strategy: Strategy for handling conflicts
            batch_config: Configuration for batch processing
            progress_callback: Callback for progress updates
            
        Returns:
            GraphStats with operation statistics
            
        Raises:
            ValidationError: If data validation fails
            DatabaseError: For database errors
            SchemaError: If relationship type is not supported
        """
        # Validate data
        valid_relationships, invalid_relationships = self._validate_data(relationships)
        
        if invalid_relationships:
            error_msg = f"{len(invalid_relationships)} invalid relationship records found"
            logger.warning(f"{error_msg}: {invalid_relationships[:5]}")
            if len(invalid_relationships) == len(relationships):
                raise ValidationError(f"All relationship records are invalid: {invalid_relationships[0][1]}")
        
        if not valid_relationships:
            logger.warning("No valid relationship records to ingest")
            return GraphStats()
        
        # Use default batch config if not provided
        config = batch_config or BatchConfig()
        
        # Group relationships by type for efficient processing
        relationships_by_type: Dict[str, List[Relationship]] = {}
        for rel in valid_relationships:
            if rel.relationship_type not in relationships_by_type:
                relationships_by_type[rel.relationship_type] = []
            relationships_by_type[rel.relationship_type].append(rel)
        
        # Process each relationship type
        combined_stats = GraphStats()
        
        for rel_type, rels in relationships_by_type.items():
            # Split data into batches
            batches = self._chunk_data(rels, config.batch_size)
            
            # Define batch processing function
            def process_batch(batch: List[Relationship]) -> GraphStats:
                # Build Cypher query based on conflict strategy
                if conflict_strategy == ConflictStrategy.MERGE:
                    query = f"""
                    UNWIND $relationships AS rel
                    MERGE (from:Address {{address: rel.from_address, chain: rel.chain}})
                    MERGE (to:Address {{address: rel.to_address, chain: rel.chain}})
                    MERGE (from)-[r:{rel_type} {{chain: rel.chain}}]->(to)
                    SET r.confidence = rel.confidence,
                        r.source = rel.source,
                        r.last_updated = datetime()
                    
                    WITH r, rel
                    FOREACH (ignoreMe IN CASE WHEN rel.properties IS NOT NULL THEN [1] ELSE [] END |
                        SET r += rel.properties
                    )
                    """
                elif conflict_strategy == ConflictStrategy.CREATE:
                    query = f"""
                    UNWIND $relationships AS rel
                    MERGE (from:Address {{address: rel.from_address, chain: rel.chain}})
                    MERGE (to:Address {{address: rel.to_address, chain: rel.chain}})
                    CREATE (from)-[r:{rel_type} {{
                        chain: rel.chain,
                        confidence: rel.confidence,
                        source: rel.source,
                        last_updated: datetime()
                    }}]->(to)
                    
                    WITH r, rel
                    FOREACH (ignoreMe IN CASE WHEN rel.properties IS NOT NULL THEN [1] ELSE [] END |
                        SET r += rel.properties
                    )
                    """
                else:  # MERGE_ON_KEY or CREATE_UNIQUE
                    query = f"""
                    UNWIND $relationships AS rel
                    MERGE (from:Address {{address: rel.from_address, chain: rel.chain}})
                    MERGE (to:Address {{address: rel.to_address, chain: rel.chain}})
                    MERGE (from)-[r:{rel_type} {{chain: rel.chain}}]->(to)
                    ON CREATE SET r.confidence = rel.confidence,
                                 r.source = rel.source,
                                 r.last_updated = datetime()
                    ON MATCH SET r.confidence = CASE
                                  WHEN r.confidence < rel.confidence THEN rel.confidence
                                  ELSE r.confidence
                                END,
                                r.last_updated = datetime()
                    
                    WITH r, rel
                    FOREACH (ignoreMe IN CASE WHEN rel.properties IS NOT NULL THEN [1] ELSE [] END |
                        SET r += rel.properties
                    )
                    """
                
                try:
                    # Convert relationships to dictionaries for Neo4j
                    rel_dicts = [r.dict() for r in batch]
                    
                    # Execute query with retry
                    start_time = time.time()
                    result = self._execute_query_with_retry(
                        query,
                        {"relationships": rel_dicts},
                        max_retries=config.retry_attempts,
                        retry_delay_seconds=config.retry_delay_seconds,
                    )
                    
                    # Process result statistics
                    stats = self._process_result_stats(result)
                    
                    # Track query time
                    stats.query_time_ms = (time.time() - start_time) * 1000
                    
                    return stats
                
                except ClientError as e:
                    # Check if the error is due to an invalid relationship type
                    if "not found" in str(e) and rel_type in str(e):
                        raise SchemaError(f"Relationship type '{rel_type}' is not defined in the database schema")
                    raise
            
            # Process batches in parallel
            if config.parallel_batches > 1:
                loop = asyncio.get_event_loop()
                type_stats = loop.run_until_complete(
                    self._process_batches_parallel(
                        batches,
                        process_batch,
                        config.parallel_batches,
                        progress_callback,
                    )
                )
            else:
                # Process batches sequentially
                type_stats = GraphStats()
                total_batches = len(batches)
                
                for i, batch in enumerate(batches):
                    stats = process_batch(batch)
                    
                    # Update type stats
                    type_stats.nodes_created += stats.nodes_created
                    type_stats.nodes_merged += stats.nodes_merged
                    type_stats.relationships_created += stats.relationships_created
                    type_stats.relationships_merged += stats.relationships_merged
                    type_stats.properties_set += stats.properties_set
                    type_stats.labels_added += stats.labels_added
                    type_stats.query_time_ms += stats.query_time_ms
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(i + 1, total_batches, (i + 1) / total_batches * 100)
                
                type_stats.batch_count = total_batches
                type_stats.total_records = len(rels)
            
            # Update combined stats
            combined_stats.nodes_created += type_stats.nodes_created
            combined_stats.nodes_merged += type_stats.nodes_merged
            combined_stats.relationships_created += type_stats.relationships_created
            combined_stats.relationships_merged += type_stats.relationships_merged
            combined_stats.properties_set += type_stats.properties_set
            combined_stats.labels_added += type_stats.labels_added
            combined_stats.query_time_ms += type_stats.query_time_ms
            combined_stats.batch_count += type_stats.batch_count
            combined_stats.total_records += type_stats.total_records
            
            logger.info(
                f"Ingested {len(rels)} {rel_type} relationships "
                f"({type_stats.relationships_created + type_stats.relationships_merged} relationships) "
                f"in {type_stats.batch_count} batches"
            )
        
        # Get a sample chain for the event
        sample_chain = valid_relationships[0].chain.value if valid_relationships else None
        
        # Publish graph event
        self._publish_graph_event(combined_stats, "relationship_ingest", sample_chain)
        
        logger.info(
            f"Ingested {len(valid_relationships)} relationships across {len(relationships_by_type)} types "
            f"({combined_stats.nodes_created + combined_stats.nodes_merged} nodes, "
            f"{combined_stats.relationships_created + combined_stats.relationships_merged} relationships) "
            f"in {combined_stats.batch_count} batches"
        )
        
        return combined_stats
    
    def execute_schema_update(self, schema_file_path: str) -> GraphStats:
        """
        Execute a schema update from a Cypher file.
        
        Args:
            schema_file_path: Path to the schema file
            
        Returns:
            GraphStats with operation statistics
            
        Raises:
            FileNotFoundError: If the schema file is not found
            SchemaError: For schema-related errors
        """
        try:
            # Read schema file
            with open(schema_file_path, "r") as f:
                schema_cypher = f.read()
            
            # Split into individual statements
            statements = [s.strip() for s in schema_cypher.split(";") if s.strip()]
            
            combined_stats = GraphStats()
            
            # Execute each statement
            for statement in statements:
                start_time = time.time()
                
                try:
                    result = self._execute_query_with_retry(statement)
                    stats = self._process_result_stats(result)
                    
                    # Track query time
                    stats.query_time_ms = (time.time() - start_time) * 1000
                    
                    # Update combined stats
                    combined_stats.nodes_created += stats.nodes_created
                    combined_stats.nodes_merged += stats.nodes_merged
                    combined_stats.relationships_created += stats.relationships_created
                    combined_stats.relationships_merged += stats.relationships_merged
                    combined_stats.properties_set += stats.properties_set
                    combined_stats.labels_added += stats.labels_added
                    combined_stats.query_time_ms += stats.query_time_ms
                
                except Exception as e:
                    logger.error(f"Error executing schema statement: {e}")
                    logger.error(f"Failed statement: {statement}")
                    raise SchemaError(f"Schema update failed: {e}")
            
            # Publish graph event
            self._publish_graph_event(combined_stats, "schema_update")
            
            logger.info(
                f"Schema update successful: {len(statements)} statements executed "
                f"({combined_stats.nodes_created + combined_stats.nodes_merged} nodes, "
                f"{combined_stats.relationships_created + combined_stats.relationships_merged} relationships)"
            )
            
            return combined_stats
        
        except FileNotFoundError:
            logger.error(f"Schema file not found: {schema_file_path}")
            raise
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the graph database.
        
        Returns:
            Dictionary with graph statistics
        """
        # Query for node counts by label
        node_query = """
        MATCH (n)
        RETURN labels(n) AS labels, count(n) AS count
        """
        
        # Query for relationship counts by type
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(r) AS count
        """
        
        # Query for chain distribution
        chain_query = """
        MATCH (n)
        WHERE n.chain IS NOT NULL
        RETURN n.chain AS chain, count(n) AS count
        """
        
        try:
            # Execute queries
            node_result = self._execute_query(node_query)
            rel_result = self._execute_query(rel_query)
            chain_result = self._execute_query(chain_query)
            
            # Process results
            node_counts = {}
            for record in node_result:
                labels = record["labels"]
                count = record["count"]
                
                # Handle multiple labels
                if len(labels) == 1:
                    node_counts[labels[0]] = count
                else:
                    # Use the combination of labels as the key
                    label_key = ":".join(sorted(labels))
                    node_counts[label_key] = count
            
            rel_counts = {}
            for record in rel_result:
                rel_type = record["type"]
                count = record["count"]
                rel_counts[rel_type] = count
            
            chain_counts = {}
            for record in chain_result:
                chain = record["chain"]
                count = record["count"]
                chain_counts[chain] = count
            
            # Calculate totals
            total_nodes = sum(node_counts.values())
            total_relationships = sum(rel_counts.values())
            
            # Return statistics
            return {
                "total_nodes": total_nodes,
                "total_relationships": total_relationships,
                "node_counts": node_counts,
                "relationship_counts": rel_counts,
                "chain_distribution": chain_counts,
                "timestamp": datetime.now().isoformat(),
            }
        
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
