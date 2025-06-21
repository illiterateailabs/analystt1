"""
Redis Client Wrapper

This module provides a comprehensive Redis client wrapper with:
- Tiered database support (DB 0 for cache, DB 1 for vector store)
- Connection pooling and management
- Automatic retry logic with exponential backoff
- Metrics integration for hit/miss rates and operation timing
- Serialization helpers for complex objects (JSON, pickle, msgpack)
- Pub/Sub capabilities for real-time updates
- TTL management and expiration policies
- Batch operations for performance
- Integration with provider registry
"""

import asyncio
import functools
import json
import logging
import pickle
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

import msgpack
import redis
from redis.client import Pipeline
from redis.exceptions import (
    ConnectionError,
    RedisError,
    ResponseError,
    TimeoutError,
    WatchError,
)
from redis.retry import Retry
from tenacity import (
    RetryError,
    Retrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from backend.core.events import CacheAddEvent, CacheHitEvent, CacheMissEvent, publish_event
from backend.core.metrics import DatabaseMetrics
from backend.providers import get_provider

# Configure module logger
logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar("T")


class RedisDb(int, Enum):
    """Redis database numbers for different purposes."""
    CACHE = 0  # Short-term cache for API responses, sessions, etc.
    VECTOR = 1  # Vector store for embeddings and graph-aware RAG


class SerializationFormat(str, Enum):
    """Serialization formats for storing complex objects."""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    RAW = "raw"  # No serialization, store as-is


class CachePolicy(str, Enum):
    """Cache expiration policies."""
    DEFAULT = "default"  # Use the default TTL
    SLIDING = "sliding"  # Reset TTL on access
    PERMANENT = "permanent"  # Never expire
    REFRESH_AHEAD = "refresh_ahead"  # Refresh before expiration


class RedisError(Exception):
    """Base exception for Redis-related errors."""
    pass


class ConnectionPoolError(RedisError):
    """Exception raised for connection pool errors."""
    pass


class SerializationError(RedisError):
    """Exception raised for serialization/deserialization errors."""
    pass


class RedisClient:
    """
    Comprehensive Redis client wrapper with tiered database support,
    connection pooling, metrics, and advanced features.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        password: Optional[str] = None,
        provider_id: str = "redis",
        default_ttl_seconds: int = 3600,
        vector_ttl_seconds: int = 86400,
        max_connections: int = 20,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 2.0,
        retry_attempts: int = 3,
        retry_backoff_min: float = 0.1,
        retry_backoff_max: float = 1.0,
        environment: str = "development",
        version: str = "1.8.0-beta",
    ):
        """
        Initialize the Redis client wrapper.
        
        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            provider_id: Provider ID for configuration
            default_ttl_seconds: Default TTL for cache entries
            vector_ttl_seconds: Default TTL for vector store entries
            max_connections: Maximum connections in the pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connection timeout in seconds
            retry_attempts: Maximum number of retry attempts
            retry_backoff_min: Minimum backoff time in seconds
            retry_backoff_max: Maximum backoff time in seconds
            environment: Environment name for metrics
            version: Application version for metrics
        """
        # Load configuration from provider registry if not provided
        if not host or not port:
            provider_config = get_provider(provider_id)
            if not provider_config:
                raise ValueError(f"Provider not found: {provider_id}")
            
            connection_uri = provider_config.get("connection_uri")
            if not connection_uri:
                raise ValueError(f"Redis URI not configured for provider: {provider_id}")
            
            # Parse connection URI
            # Format: redis://[password@]host:port
            if connection_uri.startswith("redis://"):
                uri = connection_uri[8:]  # Remove "redis://"
                
                if "@" in uri:
                    auth, hostport = uri.split("@", 1)
                    password = auth
                else:
                    hostport = uri
                
                if ":" in hostport:
                    host, port_str = hostport.split(":", 1)
                    port = int(port_str)
                else:
                    host = hostport
                    port = 6379  # Default Redis port
            
            # Get password from environment if specified
            auth_config = provider_config.get("auth", {})
            if auth_config and not password:
                import os
                password_env = auth_config.get("password_env_var")
                
                if password_env:
                    password = os.environ.get(password_env)
            
            # Get database configuration
            db_config = provider_config.get("databases", {})
            if db_config:
                cache_config = db_config.get("cache", {})
                vector_config = db_config.get("vector_store", {})
                
                if cache_config:
                    default_ttl_seconds = cache_config.get("ttl_seconds", default_ttl_seconds)
                
                if vector_config:
                    vector_ttl_seconds = vector_config.get("ttl_seconds", vector_ttl_seconds)
            
            # Get connection pool configuration
            pool_config = provider_config.get("connection_pool", {})
            if pool_config:
                max_connections = pool_config.get("max_connections", max_connections)
        
        # Store configuration
        self.host = host
        self.port = port
        self.password = password
        self.default_ttl_seconds = default_ttl_seconds
        self.vector_ttl_seconds = vector_ttl_seconds
        self.retry_attempts = retry_attempts
        self.retry_backoff_min = retry_backoff_min
        self.retry_backoff_max = retry_backoff_max
        self.environment = environment
        self.version = version
        self.provider_id = provider_id
        
        # Create connection pools for each database
        self.pools = {}
        self.clients = {}
        
        for db in RedisDb:
            # Create connection pool
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                password=password,
                db=db.value,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                retry_on_timeout=True,
                decode_responses=False,  # We'll handle decoding ourselves
            )
            
            # Create Redis client
            client = redis.Redis(connection_pool=pool)
            
            # Store pool and client
            self.pools[db] = pool
            self.clients[db] = client
        
        # Test connections
        self._test_connections()
        
        logger.info(f"Redis client initialized with host={host}, port={port}")
        logger.debug(f"Redis configuration: default_ttl={default_ttl_seconds}s, "
                    f"vector_ttl={vector_ttl_seconds}s, max_connections={max_connections}")
    
    def _test_connections(self) -> None:
        """
        Test connections to all Redis databases.
        
        Raises:
            ConnectionPoolError: If connection to any database fails
        """
        for db, client in self.clients.items():
            try:
                client.ping()
                logger.debug(f"Successfully connected to Redis DB {db.value}")
            except (ConnectionError, TimeoutError) as e:
                logger.error(f"Failed to connect to Redis DB {db.value}: {e}")
                raise ConnectionPoolError(f"Failed to connect to Redis DB {db.value}: {e}")
    
    def close(self) -> None:
        """Close all Redis connections."""
        for db, pool in self.pools.items():
            try:
                pool.disconnect()
                logger.debug(f"Disconnected from Redis DB {db.value}")
            except Exception as e:
                logger.warning(f"Error disconnecting from Redis DB {db.value}: {e}")
    
    def _get_client(self, db: RedisDb) -> redis.Redis:
        """
        Get the Redis client for the specified database.
        
        Args:
            db: Redis database enum
            
        Returns:
            Redis client for the specified database
        """
        return self.clients[db]
    
    def _serialize(self, value: Any, format: SerializationFormat) -> bytes:
        """
        Serialize a value to bytes using the specified format.
        
        Args:
            value: Value to serialize
            format: Serialization format
            
        Returns:
            Serialized value as bytes
            
        Raises:
            SerializationError: If serialization fails
        """
        try:
            if format == SerializationFormat.JSON:
                return json.dumps(value).encode("utf-8")
            elif format == SerializationFormat.PICKLE:
                return pickle.dumps(value)
            elif format == SerializationFormat.MSGPACK:
                return msgpack.packb(value)
            elif format == SerializationFormat.RAW:
                if isinstance(value, bytes):
                    return value
                elif isinstance(value, str):
                    return value.encode("utf-8")
                else:
                    raise SerializationError(f"Cannot serialize {type(value)} as RAW")
            else:
                raise SerializationError(f"Unknown serialization format: {format}")
        except Exception as e:
            logger.error(f"Serialization error ({format}): {e}")
            raise SerializationError(f"Failed to serialize value: {e}")
    
    def _deserialize(self, value: Optional[bytes], format: SerializationFormat) -> Any:
        """
        Deserialize a value from bytes using the specified format.
        
        Args:
            value: Value to deserialize
            format: Serialization format
            
        Returns:
            Deserialized value
            
        Raises:
            SerializationError: If deserialization fails
        """
        if value is None:
            return None
        
        try:
            if format == SerializationFormat.JSON:
                return json.loads(value.decode("utf-8"))
            elif format == SerializationFormat.PICKLE:
                return pickle.loads(value)
            elif format == SerializationFormat.MSGPACK:
                return msgpack.unpackb(value)
            elif format == SerializationFormat.RAW:
                return value
            else:
                raise SerializationError(f"Unknown serialization format: {format}")
        except Exception as e:
            logger.error(f"Deserialization error ({format}): {e}")
            raise SerializationError(f"Failed to deserialize value: {e}")
    
    def _with_retry(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
            
        Raises:
            RedisError: When all retries fail
        """
        try:
            for attempt in Retrying(
                stop=(
                    stop_after_attempt(self.retry_attempts) | 
                    stop_after_delay(10.0)  # Hard timeout of 10 seconds
                ),
                wait=wait_exponential(
                    multiplier=self.retry_backoff_min,
                    max=self.retry_backoff_max,
                ),
                retry=(
                    retry_if_exception_type(ConnectionError) | 
                    retry_if_exception_type(TimeoutError) |
                    retry_if_exception_type(ResponseError)
                ),
                reraise=True,
            ):
                with attempt:
                    return func(*args, **kwargs)
        except RetryError as e:
            # All retries failed
            logger.error(f"Redis operation failed after {self.retry_attempts} attempts: {e}")
            
            # Re-raise the original exception
            if e.__cause__:
                raise RedisError(f"Redis operation failed: {e.__cause__}")
            
            raise RedisError(f"Redis operation failed: {e}")
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error in Redis operation: {e}")
            raise RedisError(f"Redis operation failed: {e}")
    
    def _track_operation(
        self,
        operation: str,
        db: RedisDb,
        key: str,
        start_time: float,
        hit: bool = False,
        miss: bool = False,
    ) -> None:
        """
        Track a Redis operation with metrics and events.
        
        Args:
            operation: Operation name
            db: Redis database
            key: Key being accessed
            start_time: Operation start time
            hit: Whether this was a cache hit
            miss: Whether this was a cache miss
        """
        # Calculate operation duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Track database operation with metrics
        DatabaseMetrics.track_operation(
            database="redis",
            operation=operation,
            func=lambda: None,
            environment=self.environment,
            version=self.version,
        )()
        
        # Track operation duration
        from backend.core.metrics import db_operation_duration_seconds
        db_operation_duration_seconds.labels(
            database="redis",
            operation=operation,
            environment=self.environment,
            version=self.version,
            status="success",
        ).observe(duration_ms)
        
        # Track cache hit/miss metrics if applicable
        if hit or miss:
            # Determine cache type based on database
            cache_type = "vector" if db == RedisDb.VECTOR else "default"
            
            if hit:
                # Publish cache hit event
                publish_event(
                    event_type="data.cache_hit",
                    data={
                        "key": key,
                        "cache_type": cache_type,
                        "age_seconds": 0.0,  # We don't track age currently
                    },
                )
            
            elif miss:
                # Publish cache miss event
                publish_event(
                    event_type="data.cache_miss",
                    data={
                        "key": key,
                        "cache_type": cache_type,
                    },
                )
    
    def _track_add(
        self,
        db: RedisDb,
        key: str,
        ttl_seconds: int,
        size_bytes: int,
    ) -> None:
        """
        Track a cache add operation with events.
        
        Args:
            db: Redis database
            key: Key being added
            ttl_seconds: TTL in seconds
            size_bytes: Size of the value in bytes
        """
        # Determine cache type based on database
        cache_type = "vector" if db == RedisDb.VECTOR else "default"
        
        # Publish cache add event
        publish_event(
            event_type="data.cache_add",
            data={
                "key": key,
                "ttl_seconds": ttl_seconds,
                "size_bytes": size_bytes,
                "cache_type": cache_type,
            },
        )
    
    def get(
        self,
        key: str,
        db: RedisDb = RedisDb.CACHE,
        format: SerializationFormat = SerializationFormat.JSON,
        default: Any = None,
        policy: CachePolicy = CachePolicy.DEFAULT,
    ) -> Any:
        """
        Get a value from Redis.
        
        Args:
            key: Key to get
            db: Redis database to use
            format: Serialization format
            default: Default value if key doesn't exist
            policy: Cache policy to apply
            
        Returns:
            Deserialized value or default if key doesn't exist
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Get the value with retry
            value = self._with_retry(client.get, key)
            
            if value is None:
                # Cache miss
                self._track_operation(
                    operation="get",
                    db=db,
                    key=key,
                    start_time=start_time,
                    miss=True,
                )
                return default
            
            # Cache hit
            self._track_operation(
                operation="get",
                db=db,
                key=key,
                start_time=start_time,
                hit=True,
            )
            
            # Apply cache policy
            if policy == CachePolicy.SLIDING:
                # Reset TTL on access
                ttl = client.ttl(key)
                if ttl > 0:
                    client.expire(key, ttl)
            
            # Deserialize the value
            return self._deserialize(value, format)
        
        except Exception as e:
            logger.error(f"Error getting key {key} from Redis DB {db.value}: {e}")
            self._track_operation(
                operation="get",
                db=db,
                key=key,
                start_time=start_time,
            )
            return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        db: RedisDb = RedisDb.CACHE,
        format: SerializationFormat = SerializationFormat.JSON,
        nx: bool = False,
        xx: bool = False,
        policy: CachePolicy = CachePolicy.DEFAULT,
    ) -> bool:
        """
        Set a value in Redis.
        
        Args:
            key: Key to set
            value: Value to set
            ttl_seconds: TTL in seconds (None for default)
            db: Redis database to use
            format: Serialization format
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            policy: Cache policy to apply
            
        Returns:
            True if the value was set, False otherwise
        """
        client = self._get_client(db)
        start_time = time.time()
        
        # Determine TTL based on database and policy
        if ttl_seconds is None:
            if policy == CachePolicy.PERMANENT:
                ttl_seconds = None  # No expiration
            elif db == RedisDb.VECTOR:
                ttl_seconds = self.vector_ttl_seconds
            else:
                ttl_seconds = self.default_ttl_seconds
        
        try:
            # Serialize the value
            serialized = self._serialize(value, format)
            
            # Set the value with retry
            result = self._with_retry(
                client.set,
                key,
                serialized,
                ex=ttl_seconds,
                nx=nx,
                xx=xx,
            )
            
            self._track_operation(
                operation="set",
                db=db,
                key=key,
                start_time=start_time,
            )
            
            # Track cache add if successful
            if result:
                self._track_add(
                    db=db,
                    key=key,
                    ttl_seconds=ttl_seconds or -1,
                    size_bytes=len(serialized),
                )
            
            return bool(result)
        
        except Exception as e:
            logger.error(f"Error setting key {key} in Redis DB {db.value}: {e}")
            self._track_operation(
                operation="set",
                db=db,
                key=key,
                start_time=start_time,
            )
            return False
    
    def delete(self, key: str, db: RedisDb = RedisDb.CACHE) -> bool:
        """
        Delete a key from Redis.
        
        Args:
            key: Key to delete
            db: Redis database to use
            
        Returns:
            True if the key was deleted, False otherwise
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Delete the key with retry
            result = self._with_retry(client.delete, key)
            
            self._track_operation(
                operation="delete",
                db=db,
                key=key,
                start_time=start_time,
            )
            
            return bool(result)
        
        except Exception as e:
            logger.error(f"Error deleting key {key} from Redis DB {db.value}: {e}")
            self._track_operation(
                operation="delete",
                db=db,
                key=key,
                start_time=start_time,
            )
            return False
    
    def exists(self, key: str, db: RedisDb = RedisDb.CACHE) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: Key to check
            db: Redis database to use
            
        Returns:
            True if the key exists, False otherwise
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Check if the key exists with retry
            result = self._with_retry(client.exists, key)
            
            self._track_operation(
                operation="exists",
                db=db,
                key=key,
                start_time=start_time,
            )
            
            return bool(result)
        
        except Exception as e:
            logger.error(f"Error checking if key {key} exists in Redis DB {db.value}: {e}")
            self._track_operation(
                operation="exists",
                db=db,
                key=key,
                start_time=start_time,
            )
            return False
    
    def expire(self, key: str, ttl_seconds: int, db: RedisDb = RedisDb.CACHE) -> bool:
        """
        Set the TTL for a key.
        
        Args:
            key: Key to set TTL for
            ttl_seconds: TTL in seconds
            db: Redis database to use
            
        Returns:
            True if the TTL was set, False otherwise
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Set the TTL with retry
            result = self._with_retry(client.expire, key, ttl_seconds)
            
            self._track_operation(
                operation="expire",
                db=db,
                key=key,
                start_time=start_time,
            )
            
            return bool(result)
        
        except Exception as e:
            logger.error(f"Error setting TTL for key {key} in Redis DB {db.value}: {e}")
            self._track_operation(
                operation="expire",
                db=db,
                key=key,
                start_time=start_time,
            )
            return False
    
    def ttl(self, key: str, db: RedisDb = RedisDb.CACHE) -> int:
        """
        Get the TTL for a key.
        
        Args:
            key: Key to get TTL for
            db: Redis database to use
            
        Returns:
            TTL in seconds, -1 if the key exists but has no TTL,
            -2 if the key doesn't exist
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Get the TTL with retry
            result = self._with_retry(client.ttl, key)
            
            self._track_operation(
                operation="ttl",
                db=db,
                key=key,
                start_time=start_time,
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting TTL for key {key} in Redis DB {db.value}: {e}")
            self._track_operation(
                operation="ttl",
                db=db,
                key=key,
                start_time=start_time,
            )
            return -2
    
    def keys(self, pattern: str, db: RedisDb = RedisDb.CACHE) -> List[str]:
        """
        Get keys matching a pattern.
        
        Args:
            pattern: Pattern to match
            db: Redis database to use
            
        Returns:
            List of matching keys
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Get keys with retry
            result = self._with_retry(client.keys, pattern)
            
            self._track_operation(
                operation="keys",
                db=db,
                key=pattern,
                start_time=start_time,
            )
            
            # Convert bytes to strings
            return [k.decode("utf-8") if isinstance(k, bytes) else k for k in result]
        
        except Exception as e:
            logger.error(f"Error getting keys matching {pattern} from Redis DB {db.value}: {e}")
            self._track_operation(
                operation="keys",
                db=db,
                key=pattern,
                start_time=start_time,
            )
            return []
    
    def scan(
        self,
        cursor: int = 0,
        match: Optional[str] = None,
        count: Optional[int] = None,
        db: RedisDb = RedisDb.CACHE,
    ) -> Tuple[int, List[str]]:
        """
        Scan for keys matching a pattern.
        
        Args:
            cursor: Cursor position
            match: Pattern to match
            count: Number of keys to return
            db: Redis database to use
            
        Returns:
            Tuple of (next_cursor, keys)
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Scan keys with retry
            result = self._with_retry(
                client.scan,
                cursor,
                match=match,
                count=count,
            )
            
            self._track_operation(
                operation="scan",
                db=db,
                key=match or "*",
                start_time=start_time,
            )
            
            # Convert bytes to strings
            next_cursor = result[0]
            keys = [k.decode("utf-8") if isinstance(k, bytes) else k for k in result[1]]
            
            return next_cursor, keys
        
        except Exception as e:
            logger.error(f"Error scanning keys in Redis DB {db.value}: {e}")
            self._track_operation(
                operation="scan",
                db=db,
                key=match or "*",
                start_time=start_time,
            )
            return 0, []
    
    def scan_iter(
        self,
        match: Optional[str] = None,
        count: Optional[int] = None,
        db: RedisDb = RedisDb.CACHE,
    ) -> List[str]:
        """
        Scan for all keys matching a pattern.
        
        Args:
            match: Pattern to match
            count: Number of keys to return per scan
            db: Redis database to use
            
        Returns:
            List of matching keys
        """
        cursor = 0
        keys = []
        
        while True:
            cursor, batch = self.scan(cursor, match, count, db)
            keys.extend(batch)
            
            if cursor == 0:
                break
        
        return keys
    
    def pipeline(self, db: RedisDb = RedisDb.CACHE) -> "RedisPipeline":
        """
        Create a pipeline for batch operations.
        
        Args:
            db: Redis database to use
            
        Returns:
            Redis pipeline wrapper
        """
        client = self._get_client(db)
        pipeline = client.pipeline()
        
        return RedisPipeline(
            pipeline=pipeline,
            db=db,
            redis_client=self,
        )
    
    def flush_db(self, db: RedisDb = RedisDb.CACHE) -> bool:
        """
        Flush all keys from a database.
        
        Args:
            db: Redis database to flush
            
        Returns:
            True if the database was flushed, False otherwise
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Flush the database with retry
            result = self._with_retry(client.flushdb)
            
            self._track_operation(
                operation="flushdb",
                db=db,
                key="*",
                start_time=start_time,
            )
            
            return bool(result)
        
        except Exception as e:
            logger.error(f"Error flushing Redis DB {db.value}: {e}")
            self._track_operation(
                operation="flushdb",
                db=db,
                key="*",
                start_time=start_time,
            )
            return False
    
    def get_stats(self, db: RedisDb = RedisDb.CACHE) -> Dict[str, Any]:
        """
        Get statistics for a Redis database.
        
        Args:
            db: Redis database to get stats for
            
        Returns:
            Dictionary of statistics
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Get info with retry
            info = self._with_retry(client.info)
            
            # Get pool stats
            pool = self.pools[db]
            pool_stats = {
                "max_connections": pool.max_connections,
                "connection_count": len(pool._connections),
            }
            
            self._track_operation(
                operation="info",
                db=db,
                key="stats",
                start_time=start_time,
            )
            
            # Combine info and pool stats
            return {
                "info": info,
                "pool": pool_stats,
            }
        
        except Exception as e:
            logger.error(f"Error getting stats for Redis DB {db.value}: {e}")
            self._track_operation(
                operation="info",
                db=db,
                key="stats",
                start_time=start_time,
            )
            return {}
    
    def publish(
        self,
        channel: str,
        message: Any,
        format: SerializationFormat = SerializationFormat.JSON,
        db: RedisDb = RedisDb.CACHE,
    ) -> int:
        """
        Publish a message to a channel.
        
        Args:
            channel: Channel to publish to
            message: Message to publish
            format: Serialization format
            db: Redis database to use
            
        Returns:
            Number of clients that received the message
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Serialize the message
            serialized = self._serialize(message, format)
            
            # Publish the message with retry
            result = self._with_retry(client.publish, channel, serialized)
            
            self._track_operation(
                operation="publish",
                db=db,
                key=channel,
                start_time=start_time,
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error publishing to channel {channel} in Redis DB {db.value}: {e}")
            self._track_operation(
                operation="publish",
                db=db,
                key=channel,
                start_time=start_time,
            )
            return 0
    
    def subscribe(
        self,
        channels: List[str],
        db: RedisDb = RedisDb.CACHE,
    ) -> "PubSubWrapper":
        """
        Subscribe to channels.
        
        Args:
            channels: Channels to subscribe to
            db: Redis database to use
            
        Returns:
            PubSub wrapper
        """
        client = self._get_client(db)
        pubsub = client.pubsub()
        
        # Subscribe to channels
        pubsub.subscribe(*channels)
        
        return PubSubWrapper(
            pubsub=pubsub,
            db=db,
            redis_client=self,
        )
    
    def hget(
        self,
        name: str,
        key: str,
        db: RedisDb = RedisDb.CACHE,
        format: SerializationFormat = SerializationFormat.JSON,
        default: Any = None,
    ) -> Any:
        """
        Get a value from a hash.
        
        Args:
            name: Hash name
            key: Key in the hash
            db: Redis database to use
            format: Serialization format
            default: Default value if key doesn't exist
            
        Returns:
            Deserialized value or default if key doesn't exist
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Get the value with retry
            value = self._with_retry(client.hget, name, key)
            
            if value is None:
                # Cache miss
                self._track_operation(
                    operation="hget",
                    db=db,
                    key=f"{name}:{key}",
                    start_time=start_time,
                    miss=True,
                )
                return default
            
            # Cache hit
            self._track_operation(
                operation="hget",
                db=db,
                key=f"{name}:{key}",
                start_time=start_time,
                hit=True,
            )
            
            # Deserialize the value
            return self._deserialize(value, format)
        
        except Exception as e:
            logger.error(f"Error getting key {key} from hash {name} in Redis DB {db.value}: {e}")
            self._track_operation(
                operation="hget",
                db=db,
                key=f"{name}:{key}",
                start_time=start_time,
            )
            return default
    
    def hset(
        self,
        name: str,
        key: str,
        value: Any,
        db: RedisDb = RedisDb.CACHE,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> bool:
        """
        Set a value in a hash.
        
        Args:
            name: Hash name
            key: Key in the hash
            value: Value to set
            db: Redis database to use
            format: Serialization format
            
        Returns:
            True if the value was set, False otherwise
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Serialize the value
            serialized = self._serialize(value, format)
            
            # Set the value with retry
            result = self._with_retry(client.hset, name, key, serialized)
            
            self._track_operation(
                operation="hset",
                db=db,
                key=f"{name}:{key}",
                start_time=start_time,
            )
            
            # Track cache add if successful
            if result:
                self._track_add(
                    db=db,
                    key=f"{name}:{key}",
                    ttl_seconds=-1,  # Hashes don't have individual TTLs
                    size_bytes=len(serialized),
                )
            
            return bool(result)
        
        except Exception as e:
            logger.error(f"Error setting key {key} in hash {name} in Redis DB {db.value}: {e}")
            self._track_operation(
                operation="hset",
                db=db,
                key=f"{name}:{key}",
                start_time=start_time,
            )
            return False
    
    def hdel(self, name: str, key: str, db: RedisDb = RedisDb.CACHE) -> bool:
        """
        Delete a key from a hash.
        
        Args:
            name: Hash name
            key: Key in the hash
            db: Redis database to use
            
        Returns:
            True if the key was deleted, False otherwise
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Delete the key with retry
            result = self._with_retry(client.hdel, name, key)
            
            self._track_operation(
                operation="hdel",
                db=db,
                key=f"{name}:{key}",
                start_time=start_time,
            )
            
            return bool(result)
        
        except Exception as e:
            logger.error(f"Error deleting key {key} from hash {name} in Redis DB {db.value}: {e}")
            self._track_operation(
                operation="hdel",
                db=db,
                key=f"{name}:{key}",
                start_time=start_time,
            )
            return False
    
    def hkeys(self, name: str, db: RedisDb = RedisDb.CACHE) -> List[str]:
        """
        Get all keys in a hash.
        
        Args:
            name: Hash name
            db: Redis database to use
            
        Returns:
            List of keys in the hash
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Get keys with retry
            result = self._with_retry(client.hkeys, name)
            
            self._track_operation(
                operation="hkeys",
                db=db,
                key=name,
                start_time=start_time,
            )
            
            # Convert bytes to strings
            return [k.decode("utf-8") if isinstance(k, bytes) else k for k in result]
        
        except Exception as e:
            logger.error(f"Error getting keys from hash {name} in Redis DB {db.value}: {e}")
            self._track_operation(
                operation="hkeys",
                db=db,
                key=name,
                start_time=start_time,
            )
            return []
    
    def hgetall(
        self,
        name: str,
        db: RedisDb = RedisDb.CACHE,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> Dict[str, Any]:
        """
        Get all key-value pairs in a hash.
        
        Args:
            name: Hash name
            db: Redis database to use
            format: Serialization format
            
        Returns:
            Dictionary of key-value pairs in the hash
        """
        client = self._get_client(db)
        start_time = time.time()
        
        try:
            # Get all key-value pairs with retry
            result = self._with_retry(client.hgetall, name)
            
            self._track_operation(
                operation="hgetall",
                db=db,
                key=name,
                start_time=start_time,
                hit=bool(result),
                miss=not bool(result),
            )
            
            # Convert bytes to strings and deserialize values
            return {
                k.decode("utf-8") if isinstance(k, bytes) else k: self._deserialize(v, format)
                for k, v in result.items()
            }
        
        except Exception as e:
            logger.error(f"Error getting all key-value pairs from hash {name} in Redis DB {db.value}: {e}")
            self._track_operation(
                operation="hgetall",
                db=db,
                key=name,
                start_time=start_time,
            )
            return {}
    
    # Vector store operations
    
    def store_vector(
        self,
        key: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Store a vector in the vector store.
        
        Args:
            key: Key to store the vector under
            vector: Vector to store
            metadata: Optional metadata to store with the vector
            ttl_seconds: TTL in seconds (None for default)
            
        Returns:
            True if the vector was stored, False otherwise
        """
        # Store the vector
        vector_result = self.set(
            key=f"vector:{key}",
            value=vector,
            ttl_seconds=ttl_seconds,
            db=RedisDb.VECTOR,
            format=SerializationFormat.JSON,
        )
        
        # Store metadata if provided
        metadata_result = True
        if metadata:
            metadata_result = self.set(
                key=f"metadata:{key}",
                value=metadata,
                ttl_seconds=ttl_seconds,
                db=RedisDb.VECTOR,
                format=SerializationFormat.JSON,
            )
        
        return vector_result and metadata_result
    
    def get_vector(
        self,
        key: str,
        with_metadata: bool = False,
    ) -> Union[List[float], Tuple[List[float], Dict[str, Any]], None]:
        """
        Get a vector from the vector store.
        
        Args:
            key: Key to get the vector for
            with_metadata: Whether to include metadata
            
        Returns:
            Vector or (vector, metadata) tuple if with_metadata is True,
            or None if the vector doesn't exist
        """
        # Get the vector
        vector = self.get(
            key=f"vector:{key}",
            db=RedisDb.VECTOR,
            format=SerializationFormat.JSON,
        )
        
        if vector is None:
            return None
        
        # Get metadata if requested
        if with_metadata:
            metadata = self.get(
                key=f"metadata:{key}",
                db=RedisDb.VECTOR,
                format=SerializationFormat.JSON,
                default={},
            )
            
            return vector, metadata
        
        return vector
    
    def delete_vector(self, key: str) -> bool:
        """
        Delete a vector from the vector store.
        
        Args:
            key: Key to delete
            
        Returns:
            True if the vector was deleted, False otherwise
        """
        # Delete the vector and metadata
        vector_result = self.delete(f"vector:{key}", RedisDb.VECTOR)
        metadata_result = self.delete(f"metadata:{key}", RedisDb.VECTOR)
        
        return vector_result or metadata_result


class RedisPipeline:
    """
    Wrapper for Redis pipeline for batch operations.
    """
    
    def __init__(
        self,
        pipeline: Pipeline,
        db: RedisDb,
        redis_client: RedisClient,
    ):
        """
        Initialize the Redis pipeline wrapper.
        
        Args:
            pipeline: Redis pipeline
            db: Redis database
            redis_client: Redis client wrapper
        """
        self.pipeline = pipeline
        self.db = db
        self.redis_client = redis_client
        self.operations = []
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        format: SerializationFormat = SerializationFormat.JSON,
        nx: bool = False,
        xx: bool = False,
    ) -> "RedisPipeline":
        """
        Add a SET operation to the pipeline.
        
        Args:
            key: Key to set
            value: Value to set
            ttl_seconds: TTL in seconds (None for default)
            format: Serialization format
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            
        Returns:
            Self for chaining
        """
        # Determine TTL based on database
        if ttl_seconds is None:
            if self.db == RedisDb.VECTOR:
                ttl_seconds = self.redis_client.vector_ttl_seconds
            else:
                ttl_seconds = self.redis_client.default_ttl_seconds
        
        try:
            # Serialize the value
            serialized = self.redis_client._serialize(value, format)
            
            # Add to pipeline
            self.pipeline.set(
                key,
                serialized,
                ex=ttl_seconds,
                nx=nx,
                xx=xx,
            )
            
            # Track operation
            self.operations.append(("set", key, len(serialized)))
            
            return self
        
        except Exception as e:
            logger.error(f"Error adding SET operation for key {key} to pipeline: {e}")
            raise
    
    def get(
        self,
        key: str,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> "RedisPipeline":
        """
        Add a GET operation to the pipeline.
        
        Args:
            key: Key to get
            format: Serialization format
            
        Returns:
            Self for chaining
        """
        try:
            # Add to pipeline
            self.pipeline.get(key)
            
            # Track operation
            self.operations.append(("get", key, format))
            
            return self
        
        except Exception as e:
            logger.error(f"Error adding GET operation for key {key} to pipeline: {e}")
            raise
    
    def delete(self, key: str) -> "RedisPipeline":
        """
        Add a DELETE operation to the pipeline.
        
        Args:
            key: Key to delete
            
        Returns:
            Self for chaining
        """
        try:
            # Add to pipeline
            self.pipeline.delete(key)
            
            # Track operation
            self.operations.append(("delete", key, None))
            
            return self
        
        except Exception as e:
            logger.error(f"Error adding DELETE operation for key {key} to pipeline: {e}")
            raise
    
    def exists(self, key: str) -> "RedisPipeline":
        """
        Add an EXISTS operation to the pipeline.
        
        Args:
            key: Key to check
            
        Returns:
            Self for chaining
        """
        try:
            # Add to pipeline
            self.pipeline.exists(key)
            
            # Track operation
            self.operations.append(("exists", key, None))
            
            return self
        
        except Exception as e:
            logger.error(f"Error adding EXISTS operation for key {key} to pipeline: {e}")
            raise
    
    def expire(self, key: str, ttl_seconds: int) -> "RedisPipeline":
        """
        Add an EXPIRE operation to the pipeline.
        
        Args:
            key: Key to set TTL for
            ttl_seconds: TTL in seconds
            
        Returns:
            Self for chaining
        """
        try:
            # Add to pipeline
            self.pipeline.expire(key, ttl_seconds)
            
            # Track operation
            self.operations.append(("expire", key, ttl_seconds))
            
            return self
        
        except Exception as e:
            logger.error(f"Error adding EXPIRE operation for key {key} to pipeline: {e}")
            raise
    
    def hset(
        self,
        name: str,
        key: str,
        value: Any,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> "RedisPipeline":
        """
        Add an HSET operation to the pipeline.
        
        Args:
            name: Hash name
            key: Key in the hash
            value: Value to set
            format: Serialization format
            
        Returns:
            Self for chaining
        """
        try:
            # Serialize the value
            serialized = self.redis_client._serialize(value, format)
            
            # Add to pipeline
            self.pipeline.hset(name, key, serialized)
            
            # Track operation
            self.operations.append(("hset", f"{name}:{key}", len(serialized)))
            
            return self
        
        except Exception as e:
            logger.error(f"Error adding HSET operation for key {key} in hash {name} to pipeline: {e}")
            raise
    
    def hget(
        self,
        name: str,
        key: str,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> "RedisPipeline":
        """
        Add an HGET operation to the pipeline.
        
        Args:
            name: Hash name
            key: Key in the hash
            format: Serialization format
            
        Returns:
            Self for chaining
        """
        try:
            # Add to pipeline
            self.pipeline.hget(name, key)
            
            # Track operation
            self.operations.append(("hget", f"{name}:{key}", format))
            
            return self
        
        except Exception as e:
            logger.error(f"Error adding HGET operation for key {key} in hash {name} to pipeline: {e}")
            raise
    
    def hdel(self, name: str, key: str) -> "RedisPipeline":
        """
        Add an HDEL operation to the pipeline.
        
        Args:
            name: Hash name
            key: Key in the hash
            
        Returns:
            Self for chaining
        """
        try:
            # Add to pipeline
            self.pipeline.hdel(name, key)
            
            # Track operation
            self.operations.append(("hdel", f"{name}:{key}", None))
            
            return self
        
        except Exception as e:
            logger.error(f"Error adding HDEL operation for key {key} in hash {name} to pipeline: {e}")
            raise
    
    def execute(self) -> List[Any]:
        """
        Execute the pipeline and process the results.
        
        Returns:
            List of results for each operation
        """
        start_time = time.time()
        
        try:
            # Execute the pipeline with retry
            results = self.redis_client._with_retry(self.pipeline.execute)
            
            # Process results based on operation type
            processed_results = []
            
            for i, (op_type, key, extra) in enumerate(self.operations):
                result = results[i] if i < len(results) else None
                
                if op_type == "get" or op_type == "hget":
                    # Deserialize the value
                    format = extra
                    processed_results.append(
                        self.redis_client._deserialize(result, format) if result is not None else None
                    )
                else:
                    # Pass through other results
                    processed_results.append(result)
                
                # Track operation
                self.redis_client._track_operation(
                    operation=op_type,
                    db=self.db,
                    key=key,
                    start_time=start_time,
                    hit=op_type in ("get", "hget") and result is not None,
                    miss=op_type in ("get", "hget") and result is None,
                )
                
                # Track cache add for set operations
                if op_type == "set" and result:
                    self.redis_client._track_add(
                        db=self.db,
                        key=key,
                        ttl_seconds=self.redis_client.default_ttl_seconds,
                        size_bytes=extra,
                    )
                elif op_type == "hset" and result:
                    self.redis_client._track_add(
                        db=self.db,
                        key=key,
                        ttl_seconds=-1,  # Hashes don't have individual TTLs
                        size_bytes=extra,
                    )
            
            return processed_results
        
        except Exception as e:
            logger.error(f"Error executing pipeline: {e}")
            self.redis_client._track_operation(
                operation="pipeline",
                db=self.db,
                key="pipeline",
                start_time=start_time,
            )
            raise RedisError(f"Pipeline execution failed: {e}")
    
    def watch(self, *keys: str) -> None:
        """
        Watch keys for changes during a transaction.
        
        Args:
            *keys: Keys to watch
        """
        try:
            self.pipeline.watch(*keys)
        except Exception as e:
            logger.error(f"Error watching keys: {e}")
            raise RedisError(f"Watch failed: {e}")
    
    def unwatch(self) -> None:
        """Unwatch all keys."""
        try:
            self.pipeline.unwatch()
        except Exception as e:
            logger.error(f"Error unwatching keys: {e}")
            raise RedisError(f"Unwatch failed: {e}")
    
    def multi(self) -> "RedisPipeline":
        """
        Start a transaction.
        
        Returns:
            Self for chaining
        """
        try:
            self.pipeline.multi()
            return self
        except Exception as e:
            logger.error(f"Error starting transaction: {e}")
            raise RedisError(f"Transaction start failed: {e}")


class PubSubWrapper:
    """
    Wrapper for Redis PubSub with message processing.
    """
    
    def __init__(
        self,
        pubsub: redis.client.PubSub,
        db: RedisDb,
        redis_client: RedisClient,
    ):
        """
        Initialize the PubSub wrapper.
        
        Args:
            pubsub: Redis PubSub object
            db: Redis database
            redis_client: Redis client wrapper
        """
        self.pubsub = pubsub
        self.db = db
        self.redis_client = redis_client
        self.running = False
    
    def get_message(
        self,
        timeout: Optional[float] = None,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a message from the subscription.
        
        Args:
            timeout: Timeout in seconds
            format: Serialization format
            
        Returns:
            Message dictionary or None if no message
        """
        start_time = time.time()
        
        try:
            # Get a message with retry
            message = self.redis_client._with_retry(
                self.pubsub.get_message,
                timeout=timeout,
            )
            
            self.redis_client._track_operation(
                operation="pubsub_get",
                db=self.db,
                key="pubsub",
                start_time=start_time,
            )
            
            if not message:
                return None
            
            # Process the message
            msg_type = message.get("type")
            
            if msg_type == "message":
                # Deserialize the data
                data = message.get("data")
                if data:
                    message["data"] = self.redis_client._deserialize(data, format)
            
            return message
        
        except Exception as e:
            logger.error(f"Error getting message from PubSub: {e}")
            self.redis_client._track_operation(
                operation="pubsub_get",
                db=self.db,
                key="pubsub",
                start_time=start_time,
            )
            return None
    
    def listen(
        self,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> "PubSubListener":
        """
        Listen for messages.
        
        Args:
            format: Serialization format
            
        Returns:
            PubSub listener
        """
        return PubSubListener(
            pubsub=self.pubsub,
            db=self.db,
            redis_client=self.redis_client,
            format=format,
        )
    
    def subscribe(self, *channels: str) -> None:
        """
        Subscribe to additional channels.
        
        Args:
            *channels: Channels to subscribe to
        """
        start_time = time.time()
        
        try:
            # Subscribe with retry
            self.redis_client._with_retry(self.pubsub.subscribe, *channels)
            
            self.redis_client._track_operation(
                operation="pubsub_subscribe",
                db=self.db,
                key=",".join(channels),
                start_time=start_time,
            )
        
        except Exception as e:
            logger.error(f"Error subscribing to channels: {e}")
            self.redis_client._track_operation(
                operation="pubsub_subscribe",
                db=self.db,
                key=",".join(channels),
                start_time=start_time,
            )
            raise RedisError(f"Subscribe failed: {e}")
    
    def unsubscribe(self, *channels: str) -> None:
        """
        Unsubscribe from channels.
        
        Args:
            *channels: Channels to unsubscribe from
        """
        start_time = time.time()
        
        try:
            # Unsubscribe with retry
            self.redis_client._with_retry(self.pubsub.unsubscribe, *channels)
            
            self.redis_client._track_operation(
                operation="pubsub_unsubscribe",
                db=self.db,
                key=",".join(channels),
                start_time=start_time,
            )
        
        except Exception as e:
            logger.error(f"Error unsubscribing from channels: {e}")
            self.redis_client._track_operation(
                operation="pubsub_unsubscribe",
                db=self.db,
                key=",".join(channels),
                start_time=start_time,
            )
            raise RedisError(f"Unsubscribe failed: {e}")
    
    def close(self) -> None:
        """Close the PubSub connection."""
        try:
            self.pubsub.close()
        except Exception as e:
            logger.error(f"Error closing PubSub: {e}")


class PubSubListener:
    """
    Iterator for Redis PubSub messages.
    """
    
    def __init__(
        self,
        pubsub: redis.client.PubSub,
        db: RedisDb,
        redis_client: RedisClient,
        format: SerializationFormat = SerializationFormat.JSON,
    ):
        """
        Initialize the PubSub listener.
        
        Args:
            pubsub: Redis PubSub object
            db: Redis database
            redis_client: Redis client wrapper
            format: Serialization format
        """
        self.pubsub = pubsub
        self.db = db
        self.redis_client = redis_client
        self.format = format
        self.running = True
    
    def __iter__(self) -> "PubSubListener":
        """Return self as iterator."""
        return self
    
    def __next__(self) -> Dict[str, Any]:
        """
        Get the next message.
        
        Returns:
            Next message
            
        Raises:
            StopIteration: When stopped
        """
        if not self.running:
            raise StopIteration
        
        start_time = time.time()
        
        try:
            # Get a message
            message = self.pubsub.get_message(timeout=1.0)
            
            self.redis_client._track_operation(
                operation="pubsub_listen",
                db=self.db,
                key="pubsub",
                start_time=start_time,
            )
            
            if not message:
                # No message, try again
                return self.__next__()
            
            # Process the message
            msg_type = message.get("type")
            
            if msg_type == "message":
                # Deserialize the data
                data = message.get("data")
                if data:
                    message["data"] = self.redis_client._deserialize(data, self.format)
                
                return message
            else:
                # Skip non-message events (subscribe, etc.)
                return self.__next__()
        
        except Exception as e:
            logger.error(f"Error in PubSub listener: {e}")
            self.redis_client._track_operation(
                operation="pubsub_listen",
                db=self.db,
                key="pubsub",
                start_time=start_time,
            )
            self.running = False
            raise StopIteration
    
    def stop(self) -> None:
        """Stop the listener."""
        self.running = False
