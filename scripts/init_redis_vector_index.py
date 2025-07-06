#!/usr/bin/env python3
"""
Redis Vector Index Initialization Script

This script initializes the Redis HNSW vector index required for the Graph-RAG functionality.
It creates the necessary vector search index if it doesn't already exist.

Usage:
    python init_redis_vector_index.py [--redis-url REDIS_URL] [--dimension DIM]

Environment Variables:
    REDIS_CACHE_URL: Redis connection string (default: redis://localhost:6380/1)
    VECTOR_DIMENSION: Dimension of the vector embeddings (default: 1536)
    
Example:
    # Run with defaults
    python init_redis_vector_index.py
    
    # Specify Redis URL and dimension
    python init_redis_vector_index.py --redis-url redis://redis-cache:6379/1 --dimension 1536
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("redis-vector-init")

# Constants
DEFAULT_REDIS_URL = "redis://localhost:6380/1"  # Default to redis-cache service
DEFAULT_DIMENSION = 1536  # Default embedding dimension from graph_rag.py
INDEX_NAME = "embedding_idx"
PREFIX = "graph:"  # Prefix for keys that should be indexed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize Redis vector search index for Graph-RAG"
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default=os.getenv("REDIS_CACHE_URL", DEFAULT_REDIS_URL),
        help=f"Redis connection URL (default: {DEFAULT_REDIS_URL})",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=int(os.getenv("VECTOR_DIMENSION", DEFAULT_DIMENSION)),
        help=f"Vector dimension (default: {DEFAULT_DIMENSION})",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=os.getenv("VECTOR_INDEX_NAME", INDEX_NAME),
        help=f"Index name (default: {INDEX_NAME})",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=os.getenv("VECTOR_KEY_PREFIX", PREFIX),
        help=f"Key prefix to index (default: {PREFIX})",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Retry connection if Redis is not available",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=5,
        help="Number of connection retry attempts (default: 5)",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=2,
        help="Delay between retry attempts in seconds (default: 2)",
    )
    return parser.parse_args()


def connect_to_redis(url: str, retry: bool = False, 
                    retry_count: int = 5, retry_delay: int = 2) -> redis.Redis:
    """
    Connect to Redis with optional retry logic.
    
    Args:
        url: Redis connection URL
        retry: Whether to retry connection if it fails
        retry_count: Number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Redis client instance
        
    Raises:
        ConnectionError: If connection fails and retry is disabled or exhausted
    """
    attempt = 0
    
    while True:
        attempt += 1
        try:
            logger.info(f"Connecting to Redis at {url}")
            client = redis.from_url(url, decode_responses=True)
            # Test connection
            client.ping()
            logger.info("Successfully connected to Redis")
            return client
        except redis.ConnectionError as e:
            if not retry or attempt >= retry_count:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
            
            logger.warning(f"Connection attempt {attempt} failed. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)


def create_vector_index(
    client: redis.Redis,
    index_name: str,
    dimension: int,
    prefix: str,
    distance_metric: str = "COSINE",
) -> bool:
    """
    Create a Redis vector search index with HNSW algorithm.
    
    Args:
        client: Redis client instance
        index_name: Name of the index to create
        dimension: Dimension of the vector embeddings
        prefix: Key prefix to index
        distance_metric: Distance metric for vector search (COSINE, L2, IP)
        
    Returns:
        True if index was created, False if it already exists
    """
    try:
        # Check if index already exists
        try:
            info = client.ft(index_name).info()
            logger.info(f"Vector index '{index_name}' already exists: {info}")
            return False
        except redis.ResponseError as e:
            # Index doesn't exist, continue with creation
            if "unknown index name" not in str(e).lower():
                raise
        
        # Define the index schema
        schema = (
            # Text field for the element type
            TextField("element_type"),
            # Tag field for filtering by element ID
            TagField("element_id"),
            # Tag field for filtering by chain type
            TagField("chain"),
            # Vector field for the embedding
            VectorField(
                "embedding",
                "HNSW",  # Use HNSW algorithm for approximate nearest neighbor search
                {
                    "TYPE": "FLOAT32",  # Vector data type
                    "DIM": dimension,  # Embedding dimension
                    "DISTANCE_METRIC": distance_metric,  # Distance metric
                    "INITIAL_CAP": 1000,  # Initial vector capacity
                    "M": 16,  # Number of maximum outgoing edges per node
                    "EF_CONSTRUCTION": 200,  # Controls index build time vs search accuracy
                },
            ),
        )
        
        # Create the index
        client.ft(index_name).create_index(
            schema,
            definition=IndexDefinition(
                prefix=[prefix],  # Only index keys with this prefix
                index_type=IndexType.HASH,  # Index Redis hash data structures
            ),
        )
        
        logger.info(f"Successfully created vector index '{index_name}' with dimension {dimension}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create vector index: {e}")
        raise


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    try:
        # Connect to Redis
        client = connect_to_redis(
            args.redis_url, 
            retry=args.retry,
            retry_count=args.retry_count,
            retry_delay=args.retry_delay
        )
        
        # Create vector index
        created = create_vector_index(
            client=client,
            index_name=args.index_name,
            dimension=args.dimension,
            prefix=args.prefix,
        )
        
        # Report result
        if created:
            logger.info("Vector index initialization complete")
        else:
            logger.info("Vector index already exists, no action taken")
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Vector index initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
