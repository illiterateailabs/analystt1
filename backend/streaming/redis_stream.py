"""
Redis Streams Client - Real-time transaction streaming implementation using Redis Streams

This module provides a Redis Streams implementation for high-throughput transaction
monitoring. It includes producer and consumer functionality with support for
consumer groups, error handling, and multi-tenant isolation.

Key features:
- Stream creation and management
- Consumer groups for distributed processing
- Automatic reconnection and error handling
- Prometheus metrics for monitoring
- Multi-tenant isolation with stream prefixing
- Dead letter queue for failed messages
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

import redis
from redis.exceptions import RedisError, ConnectionError, ResponseError

from backend.core.logging import get_logger
from backend.core.metrics import REGISTRY, Counter, Histogram, Gauge

# Configure logger
logger = get_logger(__name__)

# Prometheus metrics
STREAM_MESSAGES_TOTAL = Counter(
    "stream_messages_total",
    "Total number of messages processed through Redis Streams",
    ["stream", "operation", "tenant", "status"]
)

STREAM_PROCESSING_SECONDS = Histogram(
    "stream_processing_seconds",
    "Time taken to process stream messages",
    ["stream", "operation", "tenant"]
)

STREAM_CONSUMER_LAG = Gauge(
    "stream_consumer_lag",
    "Number of messages behind in the stream",
    ["stream", "consumer_group", "tenant"]
)

STREAM_ERRORS_TOTAL = Counter(
    "stream_errors_total",
    "Total number of errors encountered in stream operations",
    ["stream", "operation", "error_type", "tenant"]
)

STREAM_ACTIVE_CONSUMERS = Gauge(
    "stream_active_consumers",
    "Number of active consumers per consumer group",
    ["stream", "consumer_group", "tenant"]
)

# Constants
DEFAULT_STREAM_PREFIX = "tx_stream"
DEFAULT_CONSUMER_GROUP = "analyst_droid_consumers"
DEFAULT_CONSUMER_NAME = "consumer"
DEFAULT_BATCH_SIZE = 100
DEFAULT_BLOCK_MS = 2000  # 2 seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_MS = 500  # 0.5 seconds
DEFAULT_CLAIM_MIN_IDLE_TIME_MS = 30000  # 30 seconds
DEFAULT_DEAD_LETTER_SUFFIX = "_dlq"
MAX_STREAM_LENGTH = 1000000  # Cap stream length to prevent memory issues


class RedisStreamClient:
    """
    Redis Streams client for high-throughput transaction streaming
    
    Provides methods for producing and consuming messages using Redis Streams,
    with support for consumer groups, error handling, and multi-tenant isolation.
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        stream_prefix: str = DEFAULT_STREAM_PREFIX,
        consumer_group: str = DEFAULT_CONSUMER_GROUP,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_ms: int = DEFAULT_RETRY_DELAY_MS
    ):
        """
        Initialize the Redis Stream client
        
        Args:
            redis_client: Redis client instance
            stream_prefix: Prefix for all stream names
            consumer_group: Default consumer group name
            max_retries: Maximum number of retry attempts for operations
            retry_delay_ms: Delay between retry attempts in milliseconds
        """
        self.redis = redis_client
        self.stream_prefix = stream_prefix
        self.consumer_group = consumer_group
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        
        logger.info(f"Initialized RedisStreamClient with prefix: {stream_prefix}")
    
    def get_stream_name(self, stream_name: str, tenant_id: Optional[str] = None) -> str:
        """
        Get the full stream name with prefix and optional tenant ID
        
        Args:
            stream_name: Base stream name
            tenant_id: Optional tenant ID for multi-tenant isolation
            
        Returns:
            Full stream name with prefix and tenant ID if provided
        """
        if tenant_id:
            return f"{self.stream_prefix}:{tenant_id}:{stream_name}"
        return f"{self.stream_prefix}:{stream_name}"
    
    def get_dead_letter_stream(self, stream_name: str) -> str:
        """
        Get the dead letter queue stream name for a given stream
        
        Args:
            stream_name: Original stream name
            
        Returns:
            Dead letter queue stream name
        """
        return f"{stream_name}{DEFAULT_DEAD_LETTER_SUFFIX}"
    
    async def create_stream(
        self,
        stream_name: str,
        tenant_id: Optional[str] = None,
        create_consumer_group: bool = True
    ) -> bool:
        """
        Create a stream and optionally a consumer group
        
        Args:
            stream_name: Base stream name
            tenant_id: Optional tenant ID for multi-tenant isolation
            create_consumer_group: Whether to create a consumer group
            
        Returns:
            True if successful, False otherwise
        """
        full_stream_name = self.get_stream_name(stream_name, tenant_id)
        
        try:
            # Add a dummy message to create the stream if it doesn't exist
            # Then delete the message to keep the stream empty
            message_id = await self._execute_with_retry(
                lambda: self.redis.xadd(full_stream_name, {"_init": "1"})
            )
            await self._execute_with_retry(
                lambda: self.redis.xdel(full_stream_name, message_id)
            )
            
            # Create consumer group if requested
            if create_consumer_group:
                try:
                    await self._execute_with_retry(
                        lambda: self.redis.xgroup_create(
                            full_stream_name, 
                            self.consumer_group, 
                            id="0", 
                            mkstream=True
                        )
                    )
                    logger.info(f"Created consumer group {self.consumer_group} for stream {full_stream_name}")
                except ResponseError as e:
                    # Ignore if group already exists
                    if "BUSYGROUP" not in str(e):
                        raise
            
            # Also create a dead letter queue stream
            dlq_stream = self.get_dead_letter_stream(full_stream_name)
            await self._execute_with_retry(
                lambda: self.redis.xadd(dlq_stream, {"_init": "1"})
            )
            await self._execute_with_retry(
                lambda: self.redis.xdel(dlq_stream, message_id)
            )
            
            logger.info(f"Successfully created stream: {full_stream_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating stream {full_stream_name}: {str(e)}")
            STREAM_ERRORS_TOTAL.labels(
                stream=stream_name,
                operation="create_stream",
                error_type=type(e).__name__,
                tenant=tenant_id or "none"
            ).inc()
            return False
    
    async def add_message(
        self,
        stream_name: str,
        message: Dict[str, Any],
        tenant_id: Optional[str] = None,
        max_len: int = MAX_STREAM_LENGTH
    ) -> Optional[str]:
        """
        Add a message to a stream
        
        Args:
            stream_name: Base stream name
            message: Dictionary containing message data
            tenant_id: Optional tenant ID for multi-tenant isolation
            max_len: Maximum length of the stream
            
        Returns:
            Message ID if successful, None otherwise
        """
        full_stream_name = self.get_stream_name(stream_name, tenant_id)
        start_time = time.time()
        
        # Convert message values to strings as required by Redis
        string_message = {k: str(v) if not isinstance(v, (str, bytes)) else v 
                         for k, v in message.items()}
        
        try:
            # Add message to the stream with approximate trimming
            message_id = await self._execute_with_retry(
                lambda: self.redis.xadd(
                    full_stream_name,
                    string_message,
                    maxlen=max_len,
                    approximate=True
                )
            )
            
            # Record metrics
            duration = time.time() - start_time
            STREAM_MESSAGES_TOTAL.labels(
                stream=stream_name,
                operation="add",
                tenant=tenant_id or "none",
                status="success"
            ).inc()
            STREAM_PROCESSING_SECONDS.labels(
                stream=stream_name,
                operation="add",
                tenant=tenant_id or "none"
            ).observe(duration)
            
            return message_id
            
        except Exception as e:
            logger.error(f"Error adding message to stream {full_stream_name}: {str(e)}")
            STREAM_ERRORS_TOTAL.labels(
                stream=stream_name,
                operation="add_message",
                error_type=type(e).__name__,
                tenant=tenant_id or "none"
            ).inc()
            STREAM_MESSAGES_TOTAL.labels(
                stream=stream_name,
                operation="add",
                tenant=tenant_id or "none",
                status="error"
            ).inc()
            return None
    
    async def add_batch(
        self,
        stream_name: str,
        messages: List[Dict[str, Any]],
        tenant_id: Optional[str] = None,
        max_len: int = MAX_STREAM_LENGTH
    ) -> int:
        """
        Add a batch of messages to a stream
        
        Args:
            stream_name: Base stream name
            messages: List of dictionaries containing message data
            tenant_id: Optional tenant ID for multi-tenant isolation
            max_len: Maximum length of the stream
            
        Returns:
            Number of successfully added messages
        """
        if not messages:
            return 0
            
        full_stream_name = self.get_stream_name(stream_name, tenant_id)
        start_time = time.time()
        success_count = 0
        
        try:
            # Use pipeline for better performance
            async with self._pipeline() as pipe:
                # Add each message to the pipeline
                for message in messages:
                    # Convert message values to strings
                    string_message = {k: str(v) if not isinstance(v, (str, bytes)) else v 
                                     for k, v in message.items()}
                    
                    pipe.xadd(
                        full_stream_name,
                        string_message,
                        maxlen=max_len,
                        approximate=True
                    )
                
                # Execute the pipeline
                results = await pipe.execute()
                success_count = len([r for r in results if r])
            
            # Record metrics
            duration = time.time() - start_time
            STREAM_MESSAGES_TOTAL.labels(
                stream=stream_name,
                operation="add_batch",
                tenant=tenant_id or "none",
                status="success"
            ).inc(success_count)
            STREAM_PROCESSING_SECONDS.labels(
                stream=stream_name,
                operation="add_batch",
                tenant=tenant_id or "none"
            ).observe(duration)
            
            if success_count < len(messages):
                logger.warning(
                    f"Only added {success_count}/{len(messages)} messages to stream {full_stream_name}"
                )
            
            return success_count
            
        except Exception as e:
            logger.error(f"Error adding batch to stream {full_stream_name}: {str(e)}")
            STREAM_ERRORS_TOTAL.labels(
                stream=stream_name,
                operation="add_batch",
                error_type=type(e).__name__,
                tenant=tenant_id or "none"
            ).inc()
            return success_count
    
    async def read_messages(
        self,
        stream_name: str,
        count: int = DEFAULT_BATCH_SIZE,
        block_ms: int = DEFAULT_BLOCK_MS,
        last_id: str = "0",
        tenant_id: Optional[str] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Read messages from a stream
        
        Args:
            stream_name: Base stream name
            count: Maximum number of messages to read
            block_ms: Time to block waiting for messages in milliseconds
            last_id: ID to start reading from
            tenant_id: Optional tenant ID for multi-tenant isolation
            
        Returns:
            List of (message_id, message) tuples
        """
        full_stream_name = self.get_stream_name(stream_name, tenant_id)
        start_time = time.time()
        
        try:
            # Read messages from the stream
            result = await self._execute_with_retry(
                lambda: self.redis.xread(
                    {full_stream_name: last_id},
                    count=count,
                    block=block_ms
                )
            )
            
            messages = []
            if result:
                # Extract messages from the result
                for stream_data in result:
                    stream, stream_messages = stream_data
                    for message_id, message in stream_messages:
                        # Decode message values
                        decoded_message = self._decode_message(message)
                        messages.append((message_id, decoded_message))
            
            # Record metrics
            duration = time.time() - start_time
            message_count = len(messages)
            STREAM_MESSAGES_TOTAL.labels(
                stream=stream_name,
                operation="read",
                tenant=tenant_id or "none",
                status="success"
            ).inc(message_count)
            STREAM_PROCESSING_SECONDS.labels(
                stream=stream_name,
                operation="read",
                tenant=tenant_id or "none"
            ).observe(duration)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error reading from stream {full_stream_name}: {str(e)}")
            STREAM_ERRORS_TOTAL.labels(
                stream=stream_name,
                operation="read_messages",
                error_type=type(e).__name__,
                tenant=tenant_id or "none"
            ).inc()
            return []
    
    async def read_group(
        self,
        stream_name: str,
        consumer_name: str = DEFAULT_CONSUMER_NAME,
        count: int = DEFAULT_BATCH_SIZE,
        block_ms: int = DEFAULT_BLOCK_MS,
        last_id: str = ">",
        tenant_id: Optional[str] = None,
        auto_acknowledge: bool = False
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Read messages from a stream using a consumer group
        
        Args:
            stream_name: Base stream name
            consumer_name: Consumer name within the group
            count: Maximum number of messages to read
            block_ms: Time to block waiting for messages in milliseconds
            last_id: ID to start reading from (> for new messages only)
            tenant_id: Optional tenant ID for multi-tenant isolation
            auto_acknowledge: Whether to automatically acknowledge messages
            
        Returns:
            List of (message_id, message) tuples
        """
        full_stream_name = self.get_stream_name(stream_name, tenant_id)
        start_time = time.time()
        
        try:
            # Ensure the stream and consumer group exist
            await self.create_stream(stream_name, tenant_id, create_consumer_group=True)
            
            # Read messages from the stream using the consumer group
            result = await self._execute_with_retry(
                lambda: self.redis.xreadgroup(
                    self.consumer_group,
                    consumer_name,
                    {full_stream_name: last_id},
                    count=count,
                    block=block_ms
                )
            )
            
            messages = []
            message_ids = []
            if result:
                # Extract messages from the result
                for stream_data in result:
                    stream, stream_messages = stream_data
                    for message_id, message in stream_messages:
                        # Decode message values
                        decoded_message = self._decode_message(message)
                        messages.append((message_id, decoded_message))
                        message_ids.append(message_id)
            
            # Auto-acknowledge messages if requested
            if auto_acknowledge and message_ids:
                await self.acknowledge_messages(stream_name, message_ids, tenant_id)
            
            # Update consumer lag metric
            await self._update_consumer_lag(stream_name, tenant_id)
            
            # Record metrics
            duration = time.time() - start_time
            message_count = len(messages)
            STREAM_MESSAGES_TOTAL.labels(
                stream=stream_name,
                operation="read_group",
                tenant=tenant_id or "none",
                status="success"
            ).inc(message_count)
            STREAM_PROCESSING_SECONDS.labels(
                stream=stream_name,
                operation="read_group",
                tenant=tenant_id or "none"
            ).observe(duration)
            
            # Update active consumers metric
            await self._update_active_consumers(stream_name, tenant_id)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error reading from consumer group on stream {full_stream_name}: {str(e)}")
            STREAM_ERRORS_TOTAL.labels(
                stream=stream_name,
                operation="read_group",
                error_type=type(e).__name__,
                tenant=tenant_id or "none"
            ).inc()
            return []
    
    async def acknowledge_messages(
        self,
        stream_name: str,
        message_ids: List[str],
        tenant_id: Optional[str] = None
    ) -> int:
        """
        Acknowledge messages in a consumer group
        
        Args:
            stream_name: Base stream name
            message_ids: List of message IDs to acknowledge
            tenant_id: Optional tenant ID for multi-tenant isolation
            
        Returns:
            Number of successfully acknowledged messages
        """
        if not message_ids:
            return 0
            
        full_stream_name = self.get_stream_name(stream_name, tenant_id)
        
        try:
            # Acknowledge messages
            result = await self._execute_with_retry(
                lambda: self.redis.xack(
                    full_stream_name,
                    self.consumer_group,
                    *message_ids
                )
            )
            
            # Also remove the messages from the pending entries list
            await self._execute_with_retry(
                lambda: self.redis.xdel(
                    full_stream_name,
                    *message_ids
                )
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error acknowledging messages in stream {full_stream_name}: {str(e)}")
            STREAM_ERRORS_TOTAL.labels(
                stream=stream_name,
                operation="acknowledge_messages",
                error_type=type(e).__name__,
                tenant=tenant_id or "none"
            ).inc()
            return 0
    
    async def move_to_dead_letter(
        self,
        stream_name: str,
        message_id: str,
        message: Dict[str, Any],
        error: str,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Move a failed message to the dead letter queue
        
        Args:
            stream_name: Base stream name
            message_id: ID of the message to move
            message: Message content
            error: Error message describing the failure
            tenant_id: Optional tenant ID for multi-tenant isolation
            
        Returns:
            True if successful, False otherwise
        """
        full_stream_name = self.get_stream_name(stream_name, tenant_id)
        dlq_stream = self.get_dead_letter_stream(full_stream_name)
        
        try:
            # Add metadata to the message
            message_with_metadata = message.copy()
            message_with_metadata.update({
                "_original_id": message_id,
                "_error": error,
                "_timestamp": str(datetime.now().isoformat()),
                "_stream": full_stream_name
            })
            
            # Add to dead letter queue
            await self._execute_with_retry(
                lambda: self.redis.xadd(
                    dlq_stream,
                    message_with_metadata
                )
            )
            
            # Acknowledge the original message
            await self._execute_with_retry(
                lambda: self.redis.xack(
                    full_stream_name,
                    self.consumer_group,
                    message_id
                )
            )
            
            # Also remove from the pending entries list
            await self._execute_with_retry(
                lambda: self.redis.xdel(
                    full_stream_name,
                    message_id
                )
            )
            
            logger.info(f"Moved message {message_id} to dead letter queue {dlq_stream}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving message to dead letter queue {dlq_stream}: {str(e)}")
            STREAM_ERRORS_TOTAL.labels(
                stream=stream_name,
                operation="move_to_dead_letter",
                error_type=type(e).__name__,
                tenant=tenant_id or "none"
            ).inc()
            return False
    
    async def claim_pending_messages(
        self,
        stream_name: str,
        consumer_name: str = DEFAULT_CONSUMER_NAME,
        min_idle_time_ms: int = DEFAULT_CLAIM_MIN_IDLE_TIME_MS,
        count: int = DEFAULT_BATCH_SIZE,
        tenant_id: Optional[str] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Claim pending messages from other consumers in the group
        
        Args:
            stream_name: Base stream name
            consumer_name: Consumer name within the group
            min_idle_time_ms: Minimum idle time in milliseconds for a message to be claimed
            count: Maximum number of messages to claim
            tenant_id: Optional tenant ID for multi-tenant isolation
            
        Returns:
            List of (message_id, message) tuples
        """
        full_stream_name = self.get_stream_name(stream_name, tenant_id)
        
        try:
            # Get pending messages
            pending = await self._execute_with_retry(
                lambda: self.redis.xpending(
                    full_stream_name,
                    self.consumer_group,
                    min_idle_time=min_idle_time_ms,
                    count=count
                )
            )
            
            if not pending:
                return []
            
            # Extract message IDs
            message_ids = [entry[0] for entry in pending]
            
            if not message_ids:
                return []
            
            # Claim the messages
            result = await self._execute_with_retry(
                lambda: self.redis.xclaim(
                    full_stream_name,
                    self.consumer_group,
                    consumer_name,
                    min_idle_time_ms,
                    message_ids,
                    justid=False
                )
            )
            
            messages = []
            for message_id, message in result:
                # Decode message values
                decoded_message = self._decode_message(message)
                messages.append((message_id, decoded_message))
            
            return messages
            
        except Exception as e:
            logger.error(f"Error claiming pending messages from stream {full_stream_name}: {str(e)}")
            STREAM_ERRORS_TOTAL.labels(
                stream=stream_name,
                operation="claim_pending_messages",
                error_type=type(e).__name__,
                tenant=tenant_id or "none"
            ).inc()
            return []
    
    async def get_stream_info(
        self,
        stream_name: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get information about a stream
        
        Args:
            stream_name: Base stream name
            tenant_id: Optional tenant ID for multi-tenant isolation
            
        Returns:
            Dictionary with stream information
        """
        full_stream_name = self.get_stream_name(stream_name, tenant_id)
        
        try:
            # Get stream information
            info = await self._execute_with_retry(
                lambda: self.redis.xinfo_stream(full_stream_name)
            )
            
            # Get consumer group information
            try:
                groups = await self._execute_with_retry(
                    lambda: self.redis.xinfo_groups(full_stream_name)
                )
            except ResponseError:
                groups = []
            
            # Format the result
            result = {
                "stream_name": full_stream_name,
                "length": info.get("length", 0),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": [
                    {
                        "name": group.get("name"),
                        "consumers": group.get("consumers", 0),
                        "pending": group.get("pending", 0),
                        "last_delivered": group.get("last-delivered-id")
                    }
                    for group in groups
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting stream info for {full_stream_name}: {str(e)}")
            STREAM_ERRORS_TOTAL.labels(
                stream=stream_name,
                operation="get_stream_info",
                error_type=type(e).__name__,
                tenant=tenant_id or "none"
            ).inc()
            return {
                "stream_name": full_stream_name,
                "error": str(e)
            }
    
    async def delete_stream(
        self,
        stream_name: str,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Delete a stream
        
        Args:
            stream_name: Base stream name
            tenant_id: Optional tenant ID for multi-tenant isolation
            
        Returns:
            True if successful, False otherwise
        """
        full_stream_name = self.get_stream_name(stream_name, tenant_id)
        
        try:
            # Delete the stream
            await self._execute_with_retry(
                lambda: self.redis.delete(full_stream_name)
            )
            
            # Delete the dead letter queue
            dlq_stream = self.get_dead_letter_stream(full_stream_name)
            await self._execute_with_retry(
                lambda: self.redis.delete(dlq_stream)
            )
            
            logger.info(f"Deleted stream: {full_stream_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting stream {full_stream_name}: {str(e)}")
            STREAM_ERRORS_TOTAL.labels(
                stream=stream_name,
                operation="delete_stream",
                error_type=type(e).__name__,
                tenant=tenant_id or "none"
            ).inc()
            return False
    
    async def _update_consumer_lag(self, stream_name: str, tenant_id: Optional[str] = None):
        """Update the consumer lag metric"""
        full_stream_name = self.get_stream_name(stream_name, tenant_id)
        
        try:
            # Get stream info
            info = await self._execute_with_retry(
                lambda: self.redis.xinfo_stream(full_stream_name)
            )
            
            # Get consumer group info
            groups = await self._execute_with_retry(
                lambda: self.redis.xinfo_groups(full_stream_name)
            )
            
            # Calculate lag for each group
            stream_length = info.get("length", 0)
            
            for group in groups:
                group_name = group.get("name")
                pending = group.get("pending", 0)
                
                # Lag is the number of messages in the stream minus the last delivered ID position
                # As an approximation, we use pending count
                lag = pending
                
                STREAM_CONSUMER_LAG.labels(
                    stream=stream_name,
                    consumer_group=group_name,
                    tenant=tenant_id or "none"
                ).set(lag)
                
        except Exception as e:
            logger.debug(f"Error updating consumer lag for {full_stream_name}: {str(e)}")
    
    async def _update_active_consumers(self, stream_name: str, tenant_id: Optional[str] = None):
        """Update the active consumers metric"""
        full_stream_name = self.get_stream_name(stream_name, tenant_id)
        
        try:
            # Get consumer group info
            groups = await self._execute_with_retry(
                lambda: self.redis.xinfo_groups(full_stream_name)
            )
            
            for group in groups:
                group_name = group.get("name")
                consumers = group.get("consumers", 0)
                
                STREAM_ACTIVE_CONSUMERS.labels(
                    stream=stream_name,
                    consumer_group=group_name,
                    tenant=tenant_id or "none"
                ).set(consumers)
                
        except Exception as e:
            logger.debug(f"Error updating active consumers for {full_stream_name}: {str(e)}")
    
    async def _execute_with_retry(self, func: Callable, retries: int = None):
        """Execute a Redis operation with retry logic"""
        if retries is None:
            retries = self.max_retries
            
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                return func()
            except (ConnectionError, TimeoutError) as e:
                last_error = e
                if attempt < retries:
                    # Exponential backoff
                    delay = (self.retry_delay_ms / 1000) * (2 ** attempt)
                    logger.warning(f"Redis operation failed, retrying in {delay:.2f}s: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Redis operation failed after {retries} retries: {str(e)}")
                    raise
            except Exception as e:
                # Don't retry other types of errors
                logger.error(f"Redis operation failed with non-retriable error: {str(e)}")
                raise
        
        if last_error:
            raise last_error
    
    @asynccontextmanager
    async def _pipeline(self):
        """Get a Redis pipeline for batch operations"""
        pipeline = self.redis.pipeline()
        try:
            yield pipeline
        finally:
            pass
    
    def _decode_message(self, message: Dict[bytes, bytes]) -> Dict[str, Any]:
        """Decode message values from bytes to Python objects"""
        result = {}
        
        for key, value in message.items():
            # Decode key
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            
            # Decode value
            if isinstance(value, bytes):
                value_str = value.decode('utf-8')
                # Try to parse JSON
                try:
                    result[key_str] = json.loads(value_str)
                except json.JSONDecodeError:
                    result[key_str] = value_str
            else:
                result[key_str] = value
        
        return result
