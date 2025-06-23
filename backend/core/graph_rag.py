"""
Graph-Aware RAG (Retrieval-Augmented Generation) Service

This module provides a comprehensive Graph-Aware RAG service that:
1. Embeds graph subgraphs into vector representations
2. Stores embeddings in Redis Vector database
3. Retrieves relevant graph context for LLM queries
4. Provides semantic search over blockchain data
5. Supports different embedding strategies (node, edge, subgraph)
6. Integrates with Neo4j for graph data
7. Supports query expansion and re-ranking
8. Provides caching and performance optimization
9. Includes comprehensive logging and metrics
10. Supports batch processing for large graphs

The service is designed to be used with LLMs to provide context-aware responses
grounded in blockchain data, with explainable evidence and citations.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, cast

import neo4j
import numpy as np
from neo4j import GraphDatabase
from pydantic import BaseModel, Field, validator
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query as RediSearchQuery

from backend.core.events import publish_event
from backend.core.metrics import ApiMetrics, DatabaseMetrics
from backend.core.neo4j_loader import Neo4jLoader
from backend.core.redis_client import RedisClient, RedisDb, SerializationFormat
from backend.providers import get_provider

# Configure module logger
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_EMBEDDING_DIMENSION = 1536  # Default for most embedding models
DEFAULT_VECTOR_SIMILARITY_THRESHOLD = 0.75
DEFAULT_CACHE_TTL_SECONDS = 3600  # 1 hour
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_CONTEXT_ITEMS = 10
DEFAULT_RERANKING_FACTOR = 0.3  # Weight for reranking (0-1)


class EmbeddingStrategy(str, Enum):
    """Strategies for embedding graph elements."""
    NODE = "node"
    EDGE = "edge"
    SUBGRAPH = "subgraph"
    PATH = "path"
    NEIGHBORHOOD = "neighborhood"


class GraphElementType(str, Enum):
    """Types of graph elements that can be embedded."""
    NODE = "node"
    RELATIONSHIP = "relationship"
    PATH = "path"
    SUBGRAPH = "subgraph"


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


class QueryExpansionStrategy(str, Enum):
    """Strategies for query expansion."""
    NONE = "none"
    SYNONYM = "synonym"
    SEMANTIC = "semantic"
    DOMAIN_SPECIFIC = "domain_specific"


class ReRankingStrategy(str, Enum):
    """Strategies for re-ranking search results."""
    NONE = "none"
    RECENCY = "recency"
    RELEVANCE = "relevance"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class GraphElement(BaseModel):
    """Base model for graph elements."""
    id: str
    type: GraphElementType
    properties: Dict[str, Any] = Field(default_factory=dict)
    labels: List[str] = Field(default_factory=list)
    chain: Optional[ChainType] = None
    
    def get_text_representation(self) -> str:
        """
        Get a text representation of the graph element for embedding.
        
        Returns:
            Text representation of the element
        """
        # Start with labels/type
        if self.labels:
            text = f"{', '.join(self.labels)} {self.id}: "
        else:
            text = f"{self.type.value} {self.id}: "
        
        # Add properties
        props = []
        for key, value in self.properties.items():
            # Skip internal properties
            if key.startswith("_"):
                continue
            
            # Format value based on type
            if isinstance(value, (list, dict)):
                value_str = str(value)
            elif isinstance(value, (int, float)):
                value_str = str(value)
            else:
                value_str = str(value)
            
            props.append(f"{key}={value_str}")
        
        text += ", ".join(props)
        
        # Add chain if available
        if self.chain:
            text += f" [chain: {self.chain.value}]"
        
        return text


class Node(GraphElement):
    """Model for a graph node."""
    type: GraphElementType = GraphElementType.NODE


class Relationship(GraphElement):
    """Model for a graph relationship."""
    type: GraphElementType = GraphElementType.RELATIONSHIP
    start_node_id: str
    end_node_id: str
    relationship_type: str
    
    def get_text_representation(self) -> str:
        """
        Get a text representation of the relationship for embedding.
        
        Returns:
            Text representation of the relationship
        """
        text = f"RELATIONSHIP {self.relationship_type}: {self.start_node_id} -> {self.end_node_id}"
        
        # Add properties
        if self.properties:
            props = []
            for key, value in self.properties.items():
                if key.startswith("_"):
                    continue
                props.append(f"{key}={value}")
            
            if props:
                text += f" {{{', '.join(props)}}}"
        
        # Add chain if available
        if self.chain:
            text += f" [chain: {self.chain.value}]"
        
        return text


class Path(GraphElement):
    """Model for a graph path."""
    type: GraphElementType = GraphElementType.PATH
    nodes: List[Node]
    relationships: List[Relationship]
    
    def get_text_representation(self) -> str:
        """
        Get a text representation of the path for embedding.
        
        Returns:
            Text representation of the path
        """
        # Format as a path: (node1)-[rel1]->(node2)-[rel2]->(node3)...
        parts = []
        
        for i, node in enumerate(self.nodes):
            # Add node
            node_text = f"({node.id}"
            if node.labels:
                node_text += f":{':'.join(node.labels)}"
            node_text += ")"
            parts.append(node_text)
            
            # Add relationship if not the last node
            if i < len(self.relationships):
                rel = self.relationships[i]
                rel_text = f"-[:{rel.relationship_type}]->"
                parts.append(rel_text)
        
        path_text = "".join(parts)
        
        # Add chain if available
        if self.chain:
            path_text += f" [chain: {self.chain.value}]"
        
        return path_text


class Subgraph(GraphElement):
    """Model for a graph subgraph."""
    type: GraphElementType = GraphElementType.SUBGRAPH
    nodes: List[Node]
    relationships: List[Relationship]
    
    def get_text_representation(self) -> str:
        """
        Get a text representation of the subgraph for embedding.
        
        Returns:
            Text representation of the subgraph
        """
        # Start with a summary
        text = f"SUBGRAPH with {len(self.nodes)} nodes and {len(self.relationships)} relationships:\n"
        
        # Add nodes
        text += "Nodes:\n"
        for i, node in enumerate(self.nodes[:5]):  # Limit to first 5 nodes
            text += f"- {node.get_text_representation()}\n"
        
        if len(self.nodes) > 5:
            text += f"... and {len(self.nodes) - 5} more nodes\n"
        
        # Add relationships
        text += "Relationships:\n"
        for i, rel in enumerate(self.relationships[:5]):  # Limit to first 5 relationships
            text += f"- {rel.get_text_representation()}\n"
        
        if len(self.relationships) > 5:
            text += f"... and {len(self.relationships) - 5} more relationships\n"
        
        # Add chain if available
        if self.chain:
            text += f"Chain: {self.chain.value}"
        
        return text


class VectorEmbedding(BaseModel):
    """Model for a vector embedding."""
    id: str
    vector: List[float]
    dimension: int
    element_id: str
    element_type: GraphElementType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @validator("dimension", always=True)
    def validate_dimension(cls, v, values):
        """Validate that dimension matches vector length."""
        if "vector" in values and v != len(values["vector"]):
            return len(values["vector"])
        return v


class SearchQuery(BaseModel):
    """Model for a search query."""
    query: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    chain: Optional[ChainType] = None
    element_types: List[GraphElementType] = Field(default_factory=lambda: [GraphElementType.NODE, GraphElementType.SUBGRAPH])
    limit: int = 10
    offset: int = 0
    min_similarity: float = 0.7
    expansion_strategy: QueryExpansionStrategy = QueryExpansionStrategy.NONE
    reranking_strategy: ReRankingStrategy = ReRankingStrategy.NONE
    include_raw_elements: bool = False


class SearchResult(BaseModel):
    """Model for a search result."""
    element_id: str
    element_type: GraphElementType
    similarity: float
    element: Optional[Union[Node, Relationship, Path, Subgraph]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResults(BaseModel):
    """Model for search results."""
    query: str
    results: List[SearchResult] = Field(default_factory=list)
    total: int = 0
    execution_time_ms: float = 0.0
    expanded_query: Optional[str] = None


class GraphContext(BaseModel):
    """Model for graph context for LLM queries."""
    elements: List[Union[Node, Relationship, Path, Subgraph]] = Field(default_factory=list)
    text_context: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    cypher_queries: List[str] = Field(default_factory=list)


class GraphRAGConfig(BaseModel):
    """Configuration for Graph RAG service."""
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    vector_similarity_threshold: float = DEFAULT_VECTOR_SIMILARITY_THRESHOLD
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    batch_size: int = DEFAULT_BATCH_SIZE
    max_context_items: int = DEFAULT_MAX_CONTEXT_ITEMS
    reranking_factor: float = DEFAULT_RERANKING_FACTOR
    embedding_provider_id: str = "gemini"
    neo4j_provider_id: str = "neo4j"
    redis_provider_id: str = "redis"


class GraphEmbedder:
    """
    Service for embedding graph elements into vector representations.
    
    This class handles the conversion of graph elements (nodes, edges, paths, subgraphs)
    into vector embeddings using various embedding strategies.
    """
    
    def __init__(
        self,
        provider_id: str = "gemini",
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize the graph embedder.
        
        Args:
            provider_id: ID of the embedding provider
            dimension: Dimension of the embeddings
            batch_size: Batch size for embedding requests
        """
        self.provider_id = provider_id
        self.dimension = dimension
        self.batch_size = batch_size
        
        # Initialize embedding provider
        self._init_embedding_provider()
    
    def _init_embedding_provider(self) -> None:
        """Initialize the embedding provider client."""
        try:
            # Get provider configuration
            provider_config = get_provider(self.provider_id)
            if not provider_config:
                raise ValueError(f"Provider not found: {self.provider_id}")
            
            # Import the appropriate client based on provider
            if self.provider_id == "gemini":
                from backend.integrations.gemini_client import GeminiClient
                self.client = GeminiClient()
                logger.info(f"Initialized embedding provider: {self.provider_id}")
            else:
                # Default to a dummy embedder for testing
                logger.warning(f"Unknown embedding provider: {self.provider_id}, using dummy embedder")
                self.client = DummyEmbedder(self.dimension)
        
        except Exception as e:
            logger.error(f"Error initializing embedding provider: {e}")
            # Fall back to dummy embedder
            self.client = DummyEmbedder(self.dimension)
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a text string into a vector.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding
        """
        start_time = time.time()
        
        try:
            # Call the embedding provider
            if hasattr(self.client, "get_embeddings"):
                vector = await self.client.get_embeddings(text)
            else:
                # Fallback for synchronous clients
                vector = self.client.get_embeddings(text)
            
            # Track metrics
            ApiMetrics.track_call(
                provider=self.provider_id,
                endpoint="embeddings",
                func=lambda: None,
                environment="development",
                version="1.8.0-beta",
            )()
            
            # Track tokens if available
            if hasattr(self.client, "count_tokens"):
                token_count = self.client.count_tokens(text)
                ApiMetrics.track_credits(
                    provider=self.provider_id,
                    endpoint="embeddings",
                    credit_type="tokens",
                    amount=token_count,
                    environment="development",
                    version="1.8.0-beta",
                )
            
            # Ensure the vector has the correct dimension
            if len(vector) != self.dimension:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )
                # Pad or truncate to match dimension
                if len(vector) < self.dimension:
                    vector = vector + [0.0] * (self.dimension - len(vector))
                else:
                    vector = vector[:self.dimension]
            
            return vector
        
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            # Return a zero vector as fallback
            return [0.0] * self.dimension
        
        finally:
            # Track duration
            duration_ms = (time.time() - start_time) * 1000
            from backend.core.metrics import external_api_duration_seconds
            external_api_duration_seconds.labels(
                provider=self.provider_id,
                endpoint="embeddings",
                status="success",
                environment="development",
                version="1.8.0-beta",
            ).observe(duration_ms / 1000)  # Convert to seconds
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple text strings into vectors.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of vector embeddings
        """
        # Process in batches to avoid overloading the API
        all_vectors = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Process batch in parallel
            tasks = [self.embed_text(text) for text in batch]
            vectors = await asyncio.gather(*tasks)
            
            all_vectors.extend(vectors)
        
        return all_vectors
    
    async def embed_element(
        self,
        element: Union[Node, Relationship, Path, Subgraph],
        strategy: EmbeddingStrategy = EmbeddingStrategy.NODE,
    ) -> VectorEmbedding:
        """
        Embed a graph element into a vector.
        
        Args:
            element: Graph element to embed
            strategy: Embedding strategy to use
            
        Returns:
            Vector embedding
        """
        # Get text representation based on strategy
        if strategy == EmbeddingStrategy.NODE and isinstance(element, Node):
            text = element.get_text_representation()
        elif strategy == EmbeddingStrategy.EDGE and isinstance(element, Relationship):
            text = element.get_text_representation()
        elif strategy == EmbeddingStrategy.PATH and isinstance(element, Path):
            text = element.get_text_representation()
        elif strategy == EmbeddingStrategy.SUBGRAPH and isinstance(element, Subgraph):
            text = element.get_text_representation()
        elif strategy == EmbeddingStrategy.NEIGHBORHOOD:
            # For neighborhood strategy, we need to create a rich text representation
            # that captures the element and its immediate connections
            if isinstance(element, Node):
                text = f"Node {element.id} with properties: {element.properties}"
                # In a real implementation, we would fetch and include neighborhood data
            else:
                text = element.get_text_representation()
        else:
            # Default to basic text representation
            text = element.get_text_representation()
        
        # Embed the text
        vector = await self.embed_text(text)
        
        # Create embedding object
        embedding_id = f"{element.type.value}_{element.id}_{strategy.value}"
        
        return VectorEmbedding(
            id=embedding_id,
            vector=vector,
            dimension=self.dimension,
            element_id=element.id,
            element_type=element.type,
            metadata={
                "strategy": strategy.value,
                "chain": element.chain.value if element.chain else None,
                "text": text[:200] + "..." if len(text) > 200 else text,  # Store truncated text
            },
        )
    
    async def embed_elements(
        self,
        elements: List[Union[Node, Relationship, Path, Subgraph]],
        strategy: EmbeddingStrategy = EmbeddingStrategy.NODE,
    ) -> List[VectorEmbedding]:
        """
        Embed multiple graph elements into vectors.
        
        Args:
            elements: List of graph elements to embed
            strategy: Embedding strategy to use
            
        Returns:
            List of vector embeddings
        """
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(elements), self.batch_size):
            batch = elements[i:i + self.batch_size]
            
            # Process batch in parallel
            tasks = [self.embed_element(element, strategy) for element in batch]
            embeddings = await asyncio.gather(*tasks)
            
            all_embeddings.extend(embeddings)
        
        return all_embeddings


class DummyEmbedder:
    """Dummy embedder for testing."""
    
    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION):
        """
        Initialize the dummy embedder.
        
        Args:
            dimension: Dimension of the embeddings
        """
        self.dimension = dimension
    
    def get_embeddings(self, text: str) -> List[float]:
        """
        Generate a deterministic embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding
        """
        # Create a deterministic hash of the text
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Use the hash to seed a random number generator
        import random
        random.seed(hash_bytes)
        
        # Generate a random vector
        vector = [random.uniform(-1, 1) for _ in range(self.dimension)]
        
        # Normalize the vector
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]
        
        return vector
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Token count
        """
        # Simple approximation: split by whitespace
        return len(text.split())


class VectorStore:
    """
    Service for storing and retrieving vector embeddings using Redis Vector Search.
    
    This class handles the storage and retrieval of vector embeddings in Redis,
    with support for similarity search and metadata filtering.
    """
    
    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        similarity_threshold: float = DEFAULT_VECTOR_SIMILARITY_THRESHOLD,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ):
        """
        Initialize the vector store.
        
        Args:
            redis_client: Redis client
            dimension: Dimension of the embeddings
            similarity_threshold: Threshold for similarity search
            cache_ttl_seconds: TTL for cached embeddings
        """
        self.redis_client = redis_client or RedisClient()
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.cache_ttl_seconds = cache_ttl_seconds
        
        self.index_name = "graph_rag_index"
        self.vector_key_prefix = "vector_embedding:"
    
    async def create_index_if_not_exists(self):
        """Create the Redis Search index if it doesn't already exist."""
        raw_client = self.redis_client._get_client(RedisDb.VECTOR)
        try:
            await raw_client.ft(self.index_name).info()
            logger.debug(f"Redis search index '{self.index_name}' already exists.")
        except Exception:
            logger.info(f"Redis search index '{self.index_name}' not found. Creating it.")
            schema = (
                VectorField(
                    "vector",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.dimension,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
                TagField("element_id"),
                TagField("element_type"),
                TagField("chain"),
                TextField("text"),
                TextField("created_at"),
                TextField("updated_at"),
                TextField("metadata_json"),
            )
            definition = IndexDefinition(prefix=[self.vector_key_prefix], index_type=IndexType.HASH)
            await raw_client.ft(self.index_name).create_index(schema, definition=definition)
            logger.info(f"Redis search index '{self.index_name}' created.")

    def _get_vector_key(self, embedding_id: str) -> str:
        """
        Get the Redis key for a vector embedding hash.
        
        Args:
            embedding_id: ID of the embedding
            
        Returns:
            Redis key
        """
        return f"{self.vector_key_prefix}{embedding_id}"
    
    async def store_embedding(self, embedding: VectorEmbedding) -> bool:
        """
        Store a vector embedding in a Redis Hash.
        
        Args:
            embedding: Vector embedding to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.create_index_if_not_exists()
            key = self._get_vector_key(embedding.id)
            vector_bytes = np.array(embedding.vector, dtype=np.float32).tobytes()

            mapping = {
                "vector": vector_bytes,
                "element_id": embedding.element_id,
                "element_type": embedding.element_type.value,
                "chain": embedding.metadata.get("chain", "unknown"),
                "text": embedding.metadata.get("text", ""),
                "created_at": embedding.created_at,
                "updated_at": embedding.updated_at,
                "metadata_json": json.dumps(embedding.metadata),
            }
            
            raw_client = self.redis_client._get_client(RedisDb.VECTOR)
            await raw_client.hset(key, mapping=mapping)
            # Note: TTL on hashes is not directly supported by FT.CREATE, managed externally if needed.
            return True
        except Exception as e:
            logger.error(f"Error storing embedding {embedding.id} in Redis hash: {e}")
            return False
    
    async def store_embeddings(self, embeddings: List[VectorEmbedding]) -> int:
        """
        Store multiple vector embeddings in Redis using a pipeline.
        
        Args:
            embeddings: List of vector embeddings to store
            
        Returns:
            Number of successfully stored embeddings
        """
        await self.create_index_if_not_exists()
        raw_client = self.redis_client._get_client(RedisDb.VECTOR)
        pipe = raw_client.pipeline(transaction=False)
        success_count = 0
        
        for embedding in embeddings:
            try:
                key = self._get_vector_key(embedding.id)
                vector_bytes = np.array(embedding.vector, dtype=np.float32).tobytes()
                mapping = {
                    "vector": vector_bytes,
                    "element_id": embedding.element_id,
                    "element_type": embedding.element_type.value,
                    "chain": embedding.metadata.get("chain", "unknown"),
                    "text": embedding.metadata.get("text", ""),
                    "created_at": embedding.created_at,
                    "updated_at": embedding.updated_at,
                    "metadata_json": json.dumps(embedding.metadata),
                }
                pipe.hset(key, mapping=mapping)
                success_count += 1
            except Exception as e:
                logger.warning(f"Could not prepare embedding {embedding.id} for pipeline: {e}")

        if success_count > 0:
            try:
                await pipe.execute()
            except Exception as e:
                logger.error(f"Error executing batch embedding pipeline: {e}")
                return 0
        return success_count
    
    async def get_embedding(self, embedding_id: str) -> Optional[VectorEmbedding]:
        """
        Get a vector embedding from a Redis Hash.
        
        Args:
            embedding_id: ID of the embedding
            
        Returns:
            Vector embedding or None if not found
        """
        try:
            key = self._get_vector_key(embedding_id)
            raw_client = self.redis_client._get_client(RedisDb.VECTOR)
            result_hash = await raw_client.hgetall(key)

            if not result_hash:
                return None

            # hgetall returns bytes, need to decode keys and some values
            decoded_hash = {k.decode('utf-8'): v for k, v in result_hash.items()}

            vector = np.frombuffer(decoded_hash['vector'], dtype=np.float32).tolist()
            metadata = json.loads(decoded_hash.get('metadata_json', '{}'))

            return VectorEmbedding(
                id=embedding_id,
                vector=vector,
                dimension=len(vector),
                element_id=decoded_hash['element_id'].decode('utf-8'),
                element_type=GraphElementType(decoded_hash['element_type'].decode('utf-8')),
                metadata=metadata,
                created_at=decoded_hash.get('created_at', b'').decode('utf-8'),
                updated_at=decoded_hash.get('updated_at', b'').decode('utf-8')
            )
        except Exception as e:
            logger.error(f"Error getting embedding {embedding_id} from hash: {e}")
            return None
    
    async def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete a vector embedding hash from Redis.
        
        Args:
            embedding_id: ID of the embedding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._get_vector_key(embedding_id)
            raw_client = self.redis_client._get_client(RedisDb.VECTOR)
            result = await raw_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting embedding hash {embedding_id}: {e}")
            return False
    
    async def search_similar(
        self,
        query_vector: List[float],
        filters: Dict[str, Any] = None,
        limit: int = 10,
        min_similarity: float = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors in Redis using FT.SEARCH.
        
        Args:
            query_vector: Query vector
            filters: Metadata filters for TAG fields
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (embedding_id, similarity, metadata) tuples
        """
        await self.create_index_if_not_exists()
        raw_client = self.redis_client._get_client(RedisDb.VECTOR)
        
        # Build filter expression
        filter_parts = []
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    # Handles lists for element_types
                    sanitized_values = [v.value if isinstance(v, Enum) else str(v) for v in value]
                    filter_parts.append(f"@{key}:{{{'|'.join(sanitized_values)}}}")
                else:
                    sanitized_value = value.value if isinstance(value, Enum) else str(value)
                    filter_parts.append(f"@{key}:{{{sanitized_value}}}")
        
        filter_str = " ".join(filter_parts) if filter_parts else "*"
        
        # Build KNN query
        query_str = f"({filter_str})=>[KNN {limit} @vector $vector_bytes AS similarity]"
        query_params = {
            "vector_bytes": np.array(query_vector, dtype=np.float32).tobytes()
        }
        
        # Create RediSearch query object
        search_query = (
            RediSearchQuery(query_str)
            .sort_by("similarity")
            .return_fields("id", "similarity", "element_id", "element_type", "metadata_json")
            .dialect(2)
        )
        
        try:
            results = await raw_client.ft(self.index_name).search(search_query, query_params)
        except Exception as e:
            logger.error(f"Redis vector search failed: {e}")
            return []

        # Parse results
        parsed_results = []
        threshold = min_similarity or self.similarity_threshold
        
        for doc in results.docs:
            # For COSINE, distance is 1 - similarity. The library returns distance.
            similarity_score = 1 - float(doc.similarity)
            
            if similarity_score >= threshold:
                metadata = json.loads(doc.metadata_json) if hasattr(doc, 'metadata_json') else {}
                embedding_id = doc.id.replace(self.vector_key_prefix, "")
                parsed_results.append((embedding_id, similarity_score, metadata))
        
        return parsed_results


class GraphQuerier:
    """
    Service for querying and traversing graph data in Neo4j.
    
    This class handles the retrieval of graph elements (nodes, edges, paths, subgraphs)
    from Neo4j, with support for various query patterns and traversal strategies.
    """
    
    def __init__(
        self,
        neo4j_loader: Optional[Neo4jLoader] = None,
        provider_id: str = "neo4j",
    ):
        """
        Initialize the graph querier.
        
        Args:
            neo4j_loader: Neo4j loader
            provider_id: Provider ID for Neo4j
        """
        self.neo4j_loader = neo4j_loader or Neo4jLoader(provider_id=provider_id)
        self.provider_id = provider_id
    
    async def get_node(self, node_id: str, labels: Optional[List[str]] = None) -> Optional[Node]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node ID
            labels: Optional node labels to filter by
            
        Returns:
            Node or None if not found
        """
        # Build Cypher query
        if labels:
            label_str = ":" + ":".join(labels)
            query = f"""
            MATCH (n{label_str} {{id: $node_id}})
            RETURN n
            """
        else:
            query = """
            MATCH (n {id: $node_id})
            RETURN n
            """
        
        # Execute query
        try:
            result = self.neo4j_loader._execute_query(query, {"node_id": node_id})
            
            # Process result
            if result.peek():
                record = result.single()
                node_data = record["n"]
                
                # Extract node properties
                properties = dict(node_data.items())
                
                # Extract node labels
                node_labels = list(node_data.labels)
                
                # Determine chain if available
                chain = None
                if "chain" in properties:
                    try:
                        chain = ChainType(properties["chain"])
                    except ValueError:
                        pass
                
                # Create Node object
                return Node(
                    id=node_id,
                    properties=properties,
                    labels=node_labels,
                    chain=chain,
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting node {node_id}: {e}")
            return None
    
    async def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """
        Get a relationship by ID.
        
        Args:
            relationship_id: Relationship ID
            
        Returns:
            Relationship or None if not found
        """
        # Build Cypher query
        query = """
        MATCH ()-[r {id: $relationship_id}]->()
        RETURN r, startNode(r) AS start, endNode(r) AS end
        """
        
        # Execute query
        try:
            result = self.neo4j_loader._execute_query(query, {"relationship_id": relationship_id})
            
            # Process result
            if result.peek():
                record = result.single()
                rel_data = record["r"]
                start_node = record["start"]
                end_node = record["end"]
                
                # Extract relationship properties
                properties = dict(rel_data.items())
                
                # Extract relationship type
                rel_type = rel_data.type
                
                # Determine chain if available
                chain = None
                if "chain" in properties:
                    try:
                        chain = ChainType(properties["chain"])
                    except ValueError:
                        pass
                
                # Create Relationship object
                return Relationship(
                    id=relationship_id,
                    properties=properties,
                    start_node_id=start_node["id"],
                    end_node_id=end_node["id"],
                    relationship_type=rel_type,
                    chain=chain,
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting relationship {relationship_id}: {e}")
            return None
    
    async def get_path(self, start_node_id: str, end_node_id: str, max_depth: int = 3) -> Optional[Path]:
        """
        Get a path between two nodes.
        
        Args:
            start_node_id: Start node ID
            end_node_id: End node ID
            max_depth: Maximum path depth
            
        Returns:
            Path or None if not found
        """
        # Build Cypher query
        query = f"""
        MATCH path = shortestPath((start {{id: $start_node_id}})-[*1..{max_depth}]-(end {{id: $end_node_id}}))
        RETURN path
        """
        
        # Execute query
        try:
            result = self.neo4j_loader._execute_query(
                query, {"start_node_id": start_node_id, "end_node_id": end_node_id}
            )
            
            # Process result
            if result.peek():
                record = result.single()
                path_data = record["path"]
                
                # Extract nodes and relationships
                nodes = []
                relationships = []
                
                # Process path nodes
                for node in path_data.nodes:
                    # Extract node properties
                    properties = dict(node.items())
                    
                    # Extract node labels
                    node_labels = list(node.labels)
                    
                    # Determine chain if available
                    chain = None
                    if "chain" in properties:
                        try:
                            chain = ChainType(properties["chain"])
                        except ValueError:
                            pass
                    
                    # Create Node object
                    nodes.append(Node(
                        id=properties.get("id", str(node.id)),
                        properties=properties,
                        labels=node_labels,
                        chain=chain,
                    ))
                
                # Process path relationships
                for rel in path_data.relationships:
                    # Extract relationship properties
                    properties = dict(rel.items())
                    
                    # Determine chain if available
                    chain = None
                    if "chain" in properties:
                        try:
                            chain = ChainType(properties["chain"])
                        except ValueError:
                            pass
                    
                    # Create Relationship object
                    relationships.append(Relationship(
                        id=properties.get("id", str(rel.id)),
                        properties=properties,
                        start_node_id=str(rel.start_node.id),
                        end_node_id=str(rel.end_node.id),
                        relationship_type=rel.type,
                        chain=chain,
                    ))
                
                # Create Path object
                path_id = f"path_{start_node_id}_{end_node_id}"
                
                # Determine chain if available (from first node)
                chain = nodes[0].chain if nodes else None
                
                return Path(
                    id=path_id,
                    nodes=nodes,
                    relationships=relationships,
                    chain=chain,
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting path from {start_node_id} to {end_node_id}: {e}")
            return None
    
    async def get_subgraph(
        self,
        center_node_id: str,
        depth: int = 1,
        relationship_types: Optional[List[str]] = None,
        node_labels: Optional[List[str]] = None,
        max_nodes: int = 100,
    ) -> Optional[Subgraph]:
        """
        Get a subgraph centered on a node.
        
        Args:
            center_node_id: Center node ID
            depth: Traversal depth
            relationship_types: Optional relationship types to filter by
            node_labels: Optional node labels to filter by
            max_nodes: Maximum number of nodes to include
            
        Returns:
            Subgraph or None if center node not found
        """
        # Build Cypher query
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(f":{rel_type}" for rel_type in relationship_types)
            rel_filter = f"[{rel_types}]"
        
        node_filter = ""
        if node_labels:
            node_labels_str = ":" + ":".join(node_labels)
            node_filter = node_labels_str
        
        query = f"""
        MATCH (center {{id: $center_node_id}})
        CALL apoc.path.subgraphAll(center, {{
            relationshipFilter: "{rel_filter}",
            labelFilter: "{node_filter}",
            maxLevel: $depth,
            limit: $max_nodes
        }})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """
        
        # Execute query
        try:
            result = self.neo4j_loader._execute_query(
                query, {
                    "center_node_id": center_node_id,
                    "depth": depth,
                    "max_nodes": max_nodes,
                }
            )
            
            # Process result
            if result.peek():
                record = result.single()
                nodes_data = record["nodes"]
                relationships_data = record["relationships"]
                
                # Process nodes
                nodes = []
                for node in nodes_data:
                    # Extract node properties
                    properties = dict(node.items())
                    
                    # Extract node labels
                    node_labels = list(node.labels)
                    
                    # Determine chain if available
                    chain = None
                    if "chain" in properties:
                        try:
                            chain = ChainType(properties["chain"])
                        except ValueError:
                            pass
                    
                    # Create Node object
                    nodes.append(Node(
                        id=properties.get("id", str(node.id)),
                        properties=properties,
                        labels=node_labels,
                        chain=chain,
                    ))
                
                # Process relationships
                relationships = []
                for rel in relationships_data:
                    # Extract relationship properties
                    properties = dict(rel.items())
                    
                    # Determine chain if available
                    chain = None
                    if "chain" in properties:
                        try:
                            chain = ChainType(properties["chain"])
                        except ValueError:
                            pass
                    
                    # Create Relationship object
                    relationships.append(Relationship(
                        id=properties.get("id", str(rel.id)),
                        properties=properties,
                        start_node_id=str(rel.start_node.id),
                        end_node_id=str(rel.end_node.id),
                        relationship_type=rel.type,
                        chain=chain,
                    ))
                
                # Create Subgraph object
                subgraph_id = f"subgraph_{center_node_id}_d{depth}"
                
                # Determine chain if available (from center node)
                center_node = next((n for n in nodes if n.id == center_node_id), None)
                chain = center_node.chain if center_node else None
                
                return Subgraph(
                    id=subgraph_id,
                    nodes=nodes,
                    relationships=relationships,
                    chain=chain,
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting subgraph for node {center_node_id}: {e}")
            return None
    
    async def execute_cypher(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return the results.
        
        Args:
            query: Cypher query
            params: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        try:
            result = self.neo4j_loader._execute_query(query, params or {})
            
            # Convert result to list of dictionaries
            records = []
            for record in result:
                record_dict = {}
                for key, value in record.items():
                    # Convert Neo4j types to Python types
                    if isinstance(value, (neo4j.graph.Node, neo4j.graph.Relationship)):
                        record_dict[key] = dict(value.items())
                    else:
                        record_dict[key] = value
                records.append(record_dict)
            
            return records
        
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return []


class QueryExpander:
    """
    Service for expanding search queries to improve recall.
    
    This class implements various query expansion strategies, such as synonym expansion,
    semantic expansion, and domain-specific expansion.
    """
    
    def __init__(self, embedder: Optional[GraphEmbedder] = None):
        """
        Initialize the query expander.
        
        Args:
            embedder: Graph embedder for semantic expansion
        """
        self.embedder = embedder
        
        # Load domain-specific synonyms
        self._load_domain_synonyms()
    
    def _load_domain_synonyms(self) -> None:
        """Load domain-specific synonyms for blockchain terms."""
        # This would typically load from a file or database
        # For now, we'll hardcode some common blockchain synonyms
        self.domain_synonyms = {
            "address": ["wallet", "account", "key"],
            "transaction": ["tx", "transfer", "payment"],
            "token": ["coin", "cryptocurrency", "asset"],
            "contract": ["smart contract", "program", "code"],
            "exchange": ["dex", "trading platform", "marketplace"],
            "wallet": ["address", "account", "purse"],
            "block": ["chunk", "segment", "batch"],
            "gas": ["fee", "cost", "price"],
            "mining": ["validation", "consensus", "proof of work"],
            "nft": ["non-fungible token", "collectible", "digital asset"],
            "defi": ["decentralized finance", "open finance", "permissionless finance"],
            "dao": ["decentralized autonomous organization", "collective", "governance"],
            "yield": ["interest", "return", "reward"],
            "staking": ["delegating", "bonding", "locking"],
            "bridge": ["cross-chain", "interoperability", "connector"],
            "liquidity": ["depth", "volume", "market depth"],
            "whale": ["large holder", "big player", "major investor"],
            "airdrop": ["distribution", "giveaway", "free tokens"],
            "rugpull": ["scam", "exit scam", "fraud"],
            "mixer": ["tumbler", "anonymizer", "privacy tool"],
        }
    
    async def expand_query(
        self,
        query: str,
        strategy: QueryExpansionStrategy = QueryExpansionStrategy.NONE,
        chain: Optional[ChainType] = None,
    ) -> str:
        """
        Expand a search query using the specified strategy.
        
        Args:
            query: Original query
            strategy: Expansion strategy
            chain: Optional blockchain type for domain-specific expansion
            
        Returns:
            Expanded query
        """
        if strategy == QueryExpansionStrategy.NONE:
            return query
        
        elif strategy == QueryExpansionStrategy.SYNONYM:
            return await self._synonym_expansion(query)
        
        elif strategy == QueryExpansionStrategy.SEMANTIC:
            return await self._semantic_expansion(query)
        
        elif strategy == QueryExpansionStrategy.DOMAIN_SPECIFIC:
            return await self._domain_expansion(query, chain)
        
        return query
    
    async def _synonym_expansion(self, query: str) -> str:
        """
        Expand query with synonyms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        expanded_terms = []
        
        # Split query into terms
        terms = query.lower().split()
        
        for term in terms:
            # Add original term
            expanded_terms.append(term)
            
            # Add synonyms if available
            if term in self.domain_synonyms:
                expanded_terms.extend(self.domain_synonyms[term])
        
        # Join terms with OR
        return " OR ".join(expanded_terms)
    
    async def _semantic_expansion(self, query: str) -> str:
        """
        Expand query semantically using embeddings.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        # This is a simplified implementation
        # In a real implementation, we would use the embedder to find semantically similar terms
        
        if not self.embedder:
            return query
        
        # For now, just add some common blockchain terms
        expanded_query = query
        
        # Add some context terms based on the query
        if "fraud" in query.lower() or "scam" in query.lower():
            expanded_query += " OR rugpull OR exit scam OR honeypot OR phishing"
        
        if "money" in query.lower() or "launder" in query.lower():
            expanded_query += " OR mixer OR tumbler OR anonymizer OR privacy"
        
        if "whale" in query.lower() or "large" in query.lower():
            expanded_query += " OR major holder OR big player OR significant investor"
        
        return expanded_query
    
    async def _domain_expansion(self, query: str, chain: Optional[ChainType] = None) -> str:
        """
        Expand query with domain-specific terms.
        
        Args:
            query: Original query
            chain: Blockchain type
            
        Returns:
            Expanded query
        """
        expanded_query = query
        
        # Add chain-specific terms if available
        if chain:
            if chain == ChainType.ETHEREUM:
                expanded_query += " ethereum ETH gwei wei ERC20 ERC721"
            elif chain == ChainType.BITCOIN:
                expanded_query += " bitcoin BTC satoshi UTXO"
            elif chain == ChainType.POLYGON:
                expanded_query += " polygon MATIC"
            elif chain == ChainType.ARBITRUM:
                expanded_query += " arbitrum ARB L2 rollup"
            elif chain == ChainType.OPTIMISM:
                expanded_query += " optimism OP L2 rollup"
            elif chain == ChainType.BASE:
                expanded_query += " base L2 rollup"
            elif chain == ChainType.SOLANA:
                expanded_query += " solana SOL"
            elif chain == ChainType.BINANCE:
                expanded_query += " binance BNB BSC"
        
        return expanded_query


class ResultReranker:
    """
    Service for re-ranking search results to improve precision.
    
    This class implements various re-ranking strategies, such as recency-based,
    relevance-based, and hybrid re-ranking.
    """
    
    def __init__(
        self,
        reranking_factor: float = DEFAULT_RERANKING_FACTOR,
    ):
        """
        Initialize the result reranker.
        
        Args:
            reranking_factor: Weight for re-ranking (0-1)
        """
        self.reranking_factor = reranking_factor
    
    async def rerank_results(
        self,
        results: List[SearchResult],
        strategy: ReRankingStrategy = ReRankingStrategy.NONE,
        query: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Re-rank search results using the specified strategy.
        
        Args:
            results: Search results to re-rank
            strategy: Re-ranking strategy
            query: Original query for relevance-based re-ranking
            
        Returns:
            Re-ranked search results
        """
        if not results or strategy == ReRankingStrategy.NONE:
            return results
        
        elif strategy == ReRankingStrategy.RECENCY:
            return await self._rerank_by_recency(results)
        
        elif strategy == ReRankingStrategy.RELEVANCE:
            return await self._rerank_by_relevance(results, query)
        
        elif strategy == ReRankingStrategy.HYBRID:
            return await self._rerank_hybrid(results, query)
        
        elif strategy == ReRankingStrategy.CUSTOM:
            return await self._rerank_custom(results, query)
        
        return results
    
    async def _rerank_by_recency(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Re-rank results by recency.
        
        Args:
            results: Search results to re-rank
            
        Returns:
            Re-ranked search results
        """
        # Extract timestamps from metadata
        for result in results:
            # Look for timestamp in metadata
            timestamp = None
            
            # Check common timestamp fields
            for field in ["timestamp", "created_at", "updated_at", "date", "time"]:
                if field in result.metadata:
                    try:
                        timestamp = datetime.fromisoformat(result.metadata[field])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Default to current time if no timestamp found
            if timestamp is None:
                timestamp = datetime.now()
            
            # Store timestamp for sorting
            result.metadata["_timestamp"] = timestamp
        
        # Get the newest and oldest timestamps
        timestamps = [r.metadata["_timestamp"] for r in results]
        newest = max(timestamps)
        oldest = min(timestamps)
        
        # Calculate time range (avoid division by zero)
        time_range = (newest - oldest).total_seconds()
        if time_range == 0:
            time_range = 1
        
        # Calculate recency score (0-1)
        for result in results:
            timestamp = result.metadata["_timestamp"]
            recency = (timestamp - oldest).total_seconds() / time_range
            
            # Combine with similarity score
            result.similarity = (1 - self.reranking_factor) * result.similarity + self.reranking_factor * recency
        
        # Sort by combined score
        results.sort(key=lambda r: r.similarity, reverse=True)
        
        return results
    
    async def _rerank_by_relevance(
        self,
        results: List[SearchResult],
        query: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Re-rank results by relevance to the query.
        
        Args:
            results: Search results to re-rank
            query: Original query
            
        Returns:
            Re-ranked search results
        """
        if not query:
            return results
        
        # This is a simplified implementation
        # In a real implementation, we would use a more sophisticated relevance model
        
        # Calculate term frequency for query terms
        query_terms = set(query.lower().split())
        
        for result in results:
            # Get text representation
            text = ""
            
            # Check for text in metadata
            if "text" in result.metadata:
                text = result.metadata["text"]
            
            # Count matching terms
            text_terms = set(text.lower().split())
            matching_terms = query_terms.intersection(text_terms)
            
            # Calculate relevance score (0-1)
            if len(query_terms) > 0:
                relevance = len(matching_terms) / len(query_terms)
            else:
                relevance = 0
            
            # Combine with similarity score
            result.similarity = (1 - self.reranking_factor) * result.similarity + self.reranking_factor * relevance
        
        # Sort by combined score
        results.sort(key=lambda r: r.similarity, reverse=True)
        
        return results
    
    async def _rerank_hybrid(
        self,
        results: List[SearchResult],
        query: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Re-rank results using a hybrid of recency and relevance.
        
        Args:
            results: Search results to re-rank
            query: Original query
            
        Returns:
            Re-ranked search results
        """
        # Apply recency re-ranking
        recency_results = await self._rerank_by_recency(results.copy())
        
        # Apply relevance re-ranking
        relevance_results = await self._rerank_by_relevance(results.copy(), query)
        
        # Combine scores
        for i, result in enumerate(results):
            recency_score = recency_results[i].similarity
            relevance_score = relevance_results[i].similarity
            
            # Equal weight for recency and relevance
            result.similarity = 0.5 * recency_score + 0.5 * relevance_score
        
        # Sort by combined score
        results.sort(key=lambda r: r.similarity, reverse=True)
        
        return results
    
    async def _rerank_custom(
        self,
        results: List[SearchResult],
        query: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Re-rank results using a custom strategy.
        
        Args:
            results: Search results to re-rank
            query: Original query
            
        Returns:
            Re-ranked search results
        """
        # This is a placeholder for a custom re-ranking strategy
        # In a real implementation, this would be tailored to specific use cases
        
        # For now, boost subgraphs and paths
        for result in results:
            if result.element_type == GraphElementType.SUBGRAPH:
                result.similarity *= 1.2  # Boost subgraphs by 20%
            elif result.element_type == GraphElementType.PATH:
                result.similarity *= 1.1  # Boost paths by 10%
            
            # Cap at 1.0
            result.similarity = min(1.0, result.similarity)
        
        # Sort by similarity
        results.sort(key=lambda r: r.similarity, reverse=True)
        
        return results


class GraphRAG:
    """
    Graph-Aware RAG (Retrieval-Augmented Generation) service.
    
    This class provides a comprehensive RAG service that integrates with Neo4j
    for graph data and Redis for vector storage, enabling semantic search over
    blockchain data and context-aware LLM responses.
    """
    
    def __init__(
        self,
        config: Optional[GraphRAGConfig] = None,
        embedder: Optional[GraphEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        graph_querier: Optional[GraphQuerier] = None,
        query_expander: Optional[QueryExpander] = None,
        result_reranker: Optional[ResultReranker] = None,
    ):
        """
        Initialize the Graph RAG service.
        
        Args:
            config: Configuration
            embedder: Graph embedder
            vector_store: Vector store
            graph_querier: Graph querier
            query_expander: Query expander
            result_reranker: Result reranker
        """
        self.config = config or GraphRAGConfig()
        
        # Initialize components
        self.embedder = embedder or GraphEmbedder(
            provider_id=self.config.embedding_provider_id,
            dimension=self.config.embedding_dimension,
            batch_size=self.config.batch_size,
        )
        
        self.vector_store = vector_store or VectorStore(
            dimension=self.config.embedding_dimension,
            similarity_threshold=self.config.vector_similarity_threshold,
            cache_ttl_seconds=self.config.cache_ttl_seconds,
        )
        
        self.graph_querier = graph_querier or GraphQuerier(
            provider_id=self.config.neo4j_provider_id,
        )
        
        self.query_expander = query_expander or QueryExpander(embedder=self.embedder)
        
        self.result_reranker = result_reranker or ResultReranker(
            reranking_factor=self.config.reranking_factor,
        )
        
        logger.info("Graph RAG service initialized")
    
    async def embed_node(
        self,
        node_id: str,
        strategy: EmbeddingStrategy = EmbeddingStrategy.NODE,
    ) -> Optional[VectorEmbedding]:
        """
        Embed a node into a vector representation.
        
        Args:
            node_id: Node ID
            strategy: Embedding strategy
            
        Returns:
            Vector embedding or None if node not found
        """
        start_time = time.time()
        
        try:
            # Get the node from Neo4j
            node = await self.graph_querier.get_node(node_id)
            if not node:
                logger.warning(f"Node not found: {node_id}")
                return None
            
            # Embed the node
            embedding = await self.embedder.embed_element(node, strategy)
            
            # Store the embedding
            await self.vector_store.store_embedding(embedding)
            
            # Track metrics
            duration_ms = (time.time() - start_time) * 1000
            DatabaseMetrics.track_operation(
                database="neo4j",
                operation="embed_node",
                func=lambda: None,
                environment="development",
                version="1.8.0-beta",
            )()
            
            # Publish event
            publish_event("graph_embedding_created", {
                "element_id": node_id,
                "element_type": GraphElementType.NODE.value,
                "strategy": strategy.value,
                "duration_ms": duration_ms,
            })
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error embedding node {node_id}: {e}")
            return None
    
    async def embed_relationship(
        self,
        relationship_id: str,
        strategy: EmbeddingStrategy = EmbeddingStrategy.EDGE,
    ) -> Optional[VectorEmbedding]:
        """
        Embed a relationship into a vector representation.
        
        Args:
            relationship_id: Relationship ID
            strategy: Embedding strategy
            
        Returns:
            Vector embedding or None if relationship not found
        """
        start_time = time.time()
        
        try:
            # Get the relationship from Neo4j
            relationship = await self.graph_querier.get_relationship(relationship_id)
            if not relationship:
                logger.warning(f"Relationship not found: {relationship_id}")
                return None
            
            # Embed the relationship
            embedding = await self.embedder.embed_element(relationship, strategy)
            
            # Store the embedding
            await self.vector_store.store_embedding(embedding)
            
            # Track metrics
            duration_ms = (time.time() - start_time) * 1000
            DatabaseMetrics.track_operation(
                database="neo4j",
                operation="embed_relationship",
                func=lambda: None,
                environment="development",
                version="1.8.0-beta",
            )()
            
            # Publish event
            publish_event("graph_embedding_created", {
                "element_id": relationship_id,
                "element_type": GraphElementType.RELATIONSHIP.value,
                "strategy": strategy.value,
                "duration_ms": duration_ms,
            })
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error embedding relationship {relationship_id}: {e}")
            return None
    
    async def embed_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 3,
        strategy: EmbeddingStrategy = EmbeddingStrategy.PATH,
    ) -> Optional[VectorEmbedding]:
        """
        Embed a path into a vector representation.
        
        Args:
            start_node_id: Start node ID
            end_node_id: End node ID
            max_depth: Maximum path depth
            strategy: Embedding strategy
            
        Returns:
            Vector embedding or None if path not found
        """
        start_time = time.time()
        
        try:
            # Get the path from Neo4j
            path = await self.graph_querier.get_path(start_node_id, end_node_id, max_depth)
            if not path:
                logger.warning(f"Path not found from {start_node_id} to {end_node_id}")
                return None
            
            # Embed the path
            embedding = await self.embedder.embed_element(path, strategy)
            
            # Store the embedding
            await self.vector_store.store_embedding(embedding)
            
            # Track metrics
            duration_ms = (time.time() - start_time) * 1000
            DatabaseMetrics.track_operation(
                database="neo4j",
                operation="embed_path",
                func=lambda: None,
                environment="development",
                version="1.8.0-beta",
            )()
            
            # Publish event
            publish_event("graph_embedding_created", {
                "element_id": path.id,
                "element_type": GraphElementType.PATH.value,
                "strategy": strategy.value,
                "duration_ms": duration_ms,
            })
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error embedding path from {start_node_id} to {end_node_id}: {e}")
            return None
    
    async def embed_subgraph(
        self,
        center_node_id: str,
        depth: int = 1,
        relationship_types: Optional[List[str]] = None,
        node_labels: Optional[List[str]] = None,
        strategy: EmbeddingStrategy = EmbeddingStrategy.SUBGRAPH,
    ) -> Optional[VectorEmbedding]:
        """
        Embed a subgraph into a vector representation.
        
        Args:
            center_node_id: Center node ID
            depth: Traversal depth
            relationship_types: Optional relationship types to filter by
            node_labels: Optional node labels to filter by
            strategy: Embedding strategy
            
        Returns:
            Vector embedding or None if subgraph not found
        """
        start_time = time.time()
        
        try:
            # Get the subgraph from Neo4j
            subgraph = await self.graph_querier.get_subgraph(
                center_node_id, depth, relationship_types, node_labels
            )
            if not subgraph:
                logger.warning(f"Subgraph not found for node {center_node_id}")
                return None
            
            # Embed the subgraph
            embedding = await self.embedder.embed_element(subgraph, strategy)
            
            # Store the embedding
            await self.vector_store.store_embedding(embedding)
            
            # Track metrics
            duration_ms = (time.time() - start_time) * 1000
            DatabaseMetrics.track_operation(
                database="neo4j",
                operation="embed_subgraph",
                func=lambda: None,
                environment="development",
                version="1.8.0-beta",
            )()
            
            # Publish event
            publish_event("graph_embedding_created", {
                "element_id": subgraph.id,
                "element_type": GraphElementType.SUBGRAPH.value,
                "strategy": strategy.value,
                "duration_ms": duration_ms,
                "node_count": len(subgraph.nodes),
                "relationship_count": len(subgraph.relationships),
            })
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error embedding subgraph for node {center_node_id}: {e}")
            return None
    
    async def batch_embed_nodes(
        self,
        node_ids: List[str],
        strategy: EmbeddingStrategy = EmbeddingStrategy.NODE,
    ) -> Dict[str, Optional[VectorEmbedding]]:
        """
        Embed multiple nodes in batch.
        
        Args:
            node_ids: List of node IDs
            strategy: Embedding strategy
            
        Returns:
            Dictionary mapping node IDs to embeddings
        """
        start_time = time.time()
        
        try:
            # Process in batches
            results = {}
            
            for i in range(0, len(node_ids), self.config.batch_size):
                batch = node_ids[i:i + self.config.batch_size]
                
                # Process batch in parallel
                tasks = [self.embed_node(node_id, strategy) for node_id in batch]
                embeddings = await asyncio.gather(*tasks)
                
                # Store results
                for j, node_id in enumerate(batch):
                    results[node_id] = embeddings[j]
            
            # Track metrics
            duration_ms = (time.time() - start_time) * 1000
            DatabaseMetrics.track_operation(
                database="neo4j",
                operation="batch_embed_nodes",
                func=lambda: None,
                environment="development",
                version="1.8.0-beta",
            )()
            
            # Publish event
            publish_event("graph_batch_embedding_created", {
                "element_type": GraphElementType.NODE.value,
                "strategy": strategy.value,
                "duration_ms": duration_ms,
                "total_count": len(node_ids),
                "success_count": sum(1 for e in results.values() if e is not None),
            })
            
            return results
        
        except Exception as e:
            logger.error(f"Error batch embedding nodes: {e}")
            return {node_id: None for node_id in node_ids}
    
    async def batch_embed_subgraphs(
        self,
        center_node_ids: List[str],
        depth: int = 1,
        relationship_types: Optional[List[str]] = None,
        node_labels: Optional[List[str]] = None,
        strategy: EmbeddingStrategy = EmbeddingStrategy.SUBGRAPH,
    ) -> Dict[str, Optional[VectorEmbedding]]:
        """
        Embed multiple subgraphs in batch.
        
        Args:
            center_node_ids: List of center node IDs
            depth: Traversal depth
            relationship_types: Optional relationship types to filter by
            node_labels: Optional node labels to filter by
            strategy: Embedding strategy
            
        Returns:
            Dictionary mapping center node IDs to embeddings
        """
        start_time = time.time()
        
        try:
            # Process in batches
            results = {}
            
            for i in range(0, len(center_node_ids), self.config.batch_size):
                batch = center_node_ids[i:i + self.config.batch_size]
                
                # Process batch in parallel
                tasks = [
                    self.embed_subgraph(
                        node_id, depth, relationship_types, node_labels, strategy
                    )
                    for node_id in batch
                ]
                embeddings = await asyncio.gather(*tasks)
                
                # Store results
                for j, node_id in enumerate(batch):
                    results[node_id] = embeddings[j]
            
            # Track metrics
            duration_ms = (time.time() - start_time) * 1000
            DatabaseMetrics.track_operation(
                database="neo4j",
                operation="batch_embed_subgraphs",
                func=lambda: None,
                environment="development",
                version="1.8.0-beta",
            )()
            
            # Publish event
            publish_event("graph_batch_embedding_created", {
                "element_type": GraphElementType.SUBGRAPH.value,
                "strategy": strategy.value,
                "duration_ms": duration_ms,
                "total_count": len(center_node_ids),
                "success_count": sum(1 for e in results.values() if e is not None),
                "depth": depth,
            })
            
            return results
        
        except Exception as e:
            logger.error(f"Error batch embedding subgraphs: {e}")
            return {node_id: None for node_id in center_node_ids}
    
    async def search(self, query: SearchQuery) -> SearchResults:
        """
        Search for graph elements similar to a query.
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        start_time = time.time()
        
        try:
            # Expand query if needed
            expanded_query = query.query
            if query.expansion_strategy != QueryExpansionStrategy.NONE:
                expanded_query = await self.query_expander.expand_query(
                    query.query, query.expansion_strategy, query.chain
                )
            
            # Embed the query
            query_vector = await self.embedder.embed_text(expanded_query)
            
            # Build filters
            filters = query.filters.copy()
            
            # Add element type filter if specified
            if query.element_types:
                filters["element_type"] = query.element_types
            
            # Add chain filter if specified
            if query.chain:
                filters["chain"] = query.chain.value
            
            # Search for similar vectors
            similar_vectors = await self.vector_store.search_similar(
                query_vector=query_vector,
                filters=filters,
                limit=query.limit + query.offset,  # Include offset in limit
                min_similarity=query.min_similarity,
            )
            
            # Apply offset
            similar_vectors = similar_vectors[query.offset:]
            
            # Create search results
            results = []
            for embedding_id, similarity, metadata in similar_vectors:
                # Get element ID and type
                element_id = metadata.get("element_id", "")
                element_type_str = metadata.get("element_type", GraphElementType.NODE.value)
                
                try:
                    element_type = GraphElementType(element_type_str)
                except ValueError:
                    element_type = GraphElementType.NODE
                
                # Create search result
                result = SearchResult(
                    element_id=element_id,
                    element_type=element_type,
                    similarity=similarity,
                    metadata=metadata,
                )
                
                # Fetch element if requested
                if query.include_raw_elements:
                    element = await self._fetch_element(element_id, element_type)
                    result.element = element
                
                results.append(result)
            
            # Re-rank results if needed
            if query.reranking_strategy != ReRankingStrategy.NONE:
                results = await self.result_reranker.rerank_results(
                    results, query.reranking_strategy, query.query
                )
            
            # Create search results
            search_results = SearchResults(
                query=query.query,
                results=results,
                total=len(similar_vectors),
                execution_time_ms=(time.time() - start_time) * 1000,
                expanded_query=expanded_query if expanded_query != query.query else None,
            )
            
            # Track metrics
            duration_ms = (time.time() - start_time) * 1000
            DatabaseMetrics.track_operation(
                database="vector",
                operation="search",
                func=lambda: None,
                environment="development",
                version="1.8.0-beta",
            )()
            
            # Publish event
            publish_event("graph_search_executed", {
                "query": query.query,
                "expanded_query": expanded_query if expanded_query != query.query else None,
                "result_count": len(results),
                "duration_ms": duration_ms,
                "min_similarity": query.min_similarity,
                "expansion_strategy": query.expansion_strategy.value,
                "reranking_strategy": query.reranking_strategy.value,
            })
            
            return search_results
        
        except Exception as e:
            logger.error(f"Error searching: {e}")
            
            # Return empty results
            return SearchResults(
                query=query.query,
                results=[],
                total=0,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
    
    async def _fetch_element(
        self,
        element_id: str,
        element_type: GraphElementType,
    ) -> Optional[Union[Node, Relationship, Path, Subgraph]]:
        """
        Fetch a graph element by ID and type.
        
        Args:
            element_id: Element ID
            element_type: Element type
            
        Returns:
            Graph element or None if not found
        """
        try:
            if element_type == GraphElementType.NODE:
                return await self.graph_querier.get_node(element_id)
            
            elif element_type == GraphElementType.RELATIONSHIP:
                return await self.graph_querier.get_relationship(element_id)
            
            elif element_type == GraphElementType.PATH:
                # Path IDs are formatted as "path_{start_node_id}_{end_node_id}"
                parts = element_id.split("_")
                if len(parts) >= 3:
                    start_node_id = parts[1]
                    end_node_id = parts[2]
                    return await self.graph_querier.get_path(start_node_id, end_node_id)
            
            elif element_type == GraphElementType.SUBGRAPH:
                # Subgraph IDs are formatted as "subgraph_{center_node_id}_d{depth}"
                parts = element_id.split("_")
                if len(parts) >= 3:
                    center_node_id = parts[1]
                    depth_part = parts[2]
                    depth = int(depth_part[1:]) if depth_part.startswith("d") else 1
                    return await self.graph_querier.get_subgraph(center_node_id, depth)
            
            return None
        
        except Exception as e:
            logger.error(f"Error fetching element {element_id} of type {element_type}:\n", e)
            return None
