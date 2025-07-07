"""
Streaming Module - Real-time transaction monitoring infrastructure

This module provides the core components for real-time blockchain transaction streaming,
enabling live monitoring, anomaly detection, and instant alerts. It supports both
Redis Streams and Kafka as backend message brokers with a unified API.

Key components:
- StreamProducer: Ingests data from blockchain sources (SIM API, etc.)
- StreamConsumer: Processes incoming stream data
- StreamProcessor: Applies transformations and detects anomalies
- WebSocketRelay: Forwards filtered stream data to frontend clients

Usage:
    from backend.streaming import setup_stream_processor, get_stream_client
    
    # Initialize the streaming infrastructure
    stream_client = get_stream_client()
    processor = setup_stream_processor(stream_client)
    
    # Start consuming from streams
    processor.start()
"""

import os
import logging
from enum import Enum
from typing import Optional, Dict, Any, Union, List

from backend.core.logging import get_logger

# Configure logger
logger = get_logger(__name__)

# Constants
DEFAULT_STREAM_PREFIX = "tx_stream"
DEFAULT_CONSUMER_GROUP = "analyst_droid_consumers"


class StreamBackend(str, Enum):
    """Supported streaming backends"""
    REDIS = "redis"
    KAFKA = "kafka"


# Determine which backend to use
STREAM_BACKEND = os.environ.get("STREAM_BACKEND", "redis").lower()
if STREAM_BACKEND not in [backend.value for backend in StreamBackend]:
    logger.warning(
        f"Invalid STREAM_BACKEND: {STREAM_BACKEND}. Defaulting to {StreamBackend.REDIS.value}"
    )
    STREAM_BACKEND = StreamBackend.REDIS.value

# Lazy-loaded clients
_stream_client = None


def get_stream_client():
    """
    Get or create the appropriate stream client based on configuration
    
    Returns:
        StreamClient instance (either RedisStreamClient or KafkaStreamClient)
    """
    global _stream_client
    
    if _stream_client is not None:
        return _stream_client
    
    if STREAM_BACKEND == StreamBackend.REDIS.value:
        # Import here to avoid circular imports
        from backend.streaming.redis_stream import RedisStreamClient
        from backend.core.redis_client import get_redis_client
        
        redis_client = get_redis_client()
        _stream_client = RedisStreamClient(redis_client)
        logger.info("Initialized Redis Stream client")
    
    elif STREAM_BACKEND == StreamBackend.KAFKA.value:
        # Import here to avoid circular imports
        from backend.streaming.kafka_stream import KafkaStreamClient
        
        kafka_bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        _stream_client = KafkaStreamClient(bootstrap_servers=kafka_bootstrap_servers)
        logger.info(f"Initialized Kafka Stream client with bootstrap servers: {kafka_bootstrap_servers}")
    
    else:
        raise ValueError(f"Unsupported stream backend: {STREAM_BACKEND}")
    
    return _stream_client


def setup_stream_processor(stream_client=None, tenant_id: Optional[str] = None):
    """
    Set up a stream processor for transaction monitoring
    
    Args:
        stream_client: Optional stream client instance (created if not provided)
        tenant_id: Optional tenant ID for multi-tenant deployments
        
    Returns:
        StreamProcessor instance ready to start consuming
    """
    # Import here to avoid circular imports
    from backend.streaming.processor import StreamProcessor
    
    if stream_client is None:
        stream_client = get_stream_client()
    
    processor = StreamProcessor(
        stream_client=stream_client,
        tenant_id=tenant_id
    )
    
    return processor


# Initialize on module import
logger.info(f"Streaming module initialized with backend: {STREAM_BACKEND}")
