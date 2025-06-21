"""
Graph-Aware RAG Tool for CrewAI Agents

This module provides a comprehensive Graph-Aware RAG (Retrieval Augmented Generation)
tool for CrewAI agents. It enables semantic search over blockchain graph data,
context enrichment, and embedding management.

Features:
1. Semantic search over blockchain graph data
2. Context enrichment for agent responses
3. Embedding storage and retrieval
4. Query expansion and context building
5. Blockchain-specific search methods
6. Integration with Redis for caching
7. Comprehensive error handling and logging
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, validator

from backend.agents.tools.base_tool import AbstractApiTool, ApiError
from backend.core.graph_rag import (
    GraphRAG,
    SearchQuery,
    GraphElementType,
    EmbeddingStrategy,
    VectorEmbedding,
    SearchResults,
    QueryExpansionStrategy,
    ReRankingStrategy,
    ChainType,
)
from backend.core.redis_client import RedisClient, RedisDb, SerializationFormat
from backend.core.telemetry import trace_async_function

# Configure module logger
logger = logging.getLogger(__name__)


class GraphRagSearchRequest(BaseModel):
    """Request model for Graph-RAG semantic search."""
    query: str = Field(..., description="The natural language query for semantic search.")
    element_types: List[GraphElementType] = Field(
        default_factory=lambda: [GraphElementType.NODE, GraphElementType.SUBGRAPH],
        description="List of graph element types to search for (e.g., NODE, RELATIONSHIP, SUBGRAPH)."
    )
    limit: int = Field(10, description="Maximum number of search results to return.")
    min_similarity: float = Field(0.7, description="Minimum similarity score for results (0.0-1.0).")
    include_raw_elements: bool = Field(False, description="Whether to include the raw graph elements in the results.")
    expansion_strategy: QueryExpansionStrategy = Field(
        QueryExpansionStrategy.NONE, description="Strategy for query expansion (e.g., SYNONYM, SEMANTIC)."
    )
    reranking_strategy: ReRankingStrategy = Field(
        ReRankingStrategy.NONE, description="Strategy for re-ranking search results (e.g., RECENCY, RELEVANCE)."
    )
    chain: Optional[ChainType] = Field(None, description="Optional blockchain chain to filter results by.")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional metadata filters for search.")


class GraphRagEmbedRequest(BaseModel):
    """Request model for Graph-RAG embedding operations."""
    element_id: str = Field(..., description="The ID of the graph element to embed.")
    element_type: GraphElementType = Field(..., description="The type of the graph element (NODE, RELATIONSHIP, SUBGRAPH, PATH).")
    strategy: EmbeddingStrategy = Field(
        EmbeddingStrategy.NODE, description="The embedding strategy to use (e.g., NODE, SUBGRAPH, PATH)."
    )
    # Optional parameters for specific element types
    chain: Optional[ChainType] = Field(None, description="Blockchain chain of the element.")
    depth: Optional[int] = Field(None, description="Traversal depth for subgraph/path embedding.")
    start_node_id: Optional[str] = Field(None, description="Start node ID for path embedding.")
    end_node_id: Optional[str] = Field(None, description="End node ID for path embedding.")


class GraphRagContextRequest(BaseModel):
    """Request model for retrieving context for a specific topic."""
    topic: str = Field(..., description="The topic to get context for (e.g., 'ethereum', 'address:0x123', 'transaction:0xabc').")
    max_results: int = Field(5, description="Maximum number of context items to return.")
    include_raw_data: bool = Field(False, description="Whether to include raw graph data in the response.")
    chain: Optional[ChainType] = Field(None, description="Optional blockchain chain to filter results by.")


class GraphRagToolRequest(BaseModel):
    """Main request model for the Graph-RAG Tool, supporting multiple operations."""
    operation: str = Field(..., description="The operation to perform: 'search', 'embed', or 'get_context'.")
    search_params: Optional[GraphRagSearchRequest] = Field(None, description="Parameters for a 'search' operation.")
    embed_params: Optional[GraphRagEmbedRequest] = Field(None, description="Parameters for an 'embed' operation.")
    context_params: Optional[GraphRagContextRequest] = Field(None, description="Parameters for a 'get_context' operation.")

    @validator('operation')
    def validate_operation(cls, v):
        if v not in ['search', 'embed', 'get_context']:
            raise ValueError("Operation must be 'search', 'embed', or 'get_context'.")
        return v

    @validator('search_params')
    def check_search_params(cls, v, values):
        if values.get('operation') == 'search' and v is None:
            raise ValueError("search_params must be provided for 'search' operation.")
        return v

    @validator('embed_params')
    def check_embed_params(cls, v, values):
        if values.get('operation') == 'embed' and v is None:
            raise ValueError("embed_params must be provided for 'embed' operation.")
        return v

    @validator('context_params')
    def check_context_params(cls, v, values):
        if values.get('operation') == 'get_context' and v is None:
            raise ValueError("context_params must be provided for 'get_context' operation.")
        return v


class GraphRagTool(AbstractApiTool):
    """
    A comprehensive Graph-Aware RAG tool for CrewAI agents.
    Provides semantic search, context enrichment, and embedding capabilities
    over blockchain graph data.
    """
    name = "graph_rag_tool"
    description = "Provides graph-aware retrieval-augmented generation (RAG) capabilities, including semantic search over blockchain graph data and embedding of graph elements."
    provider_id = "internal_graph_rag"  # Internal provider, not an external API
    request_model = GraphRagToolRequest

    def __init__(self, graph_rag_service: Optional[GraphRAG] = None, redis_client: Optional[RedisClient] = None):
        """
        Initializes the GraphRagTool.
        Args:
            graph_rag_service: An instance of the GraphRAG service. If None, a new one will be created.
            redis_client: An instance of the RedisClient. If None, a new one will be created.
        """
        super().__init__(provider_id=self.provider_id)
        self.graph_rag_service = graph_rag_service or GraphRAG()
        self.redis_client = redis_client or RedisClient()  # Used for caching if needed, though GraphRAG handles its own.
        self.client = self.graph_rag_service  # Required by AbstractApiTool

    @trace_async_function(span_name="graph_rag_tool.execute")
    async def _execute(self, request: GraphRagToolRequest) -> Union[Dict[str, Any], SearchResults, VectorEmbedding, List[VectorEmbedding]]:
        """
        Executes the requested Graph-RAG operation.
        Args:
            request: An instance of GraphRagToolRequest specifying the operation and parameters.
        Returns:
            The result of the operation (SearchResults for 'search', VectorEmbedding for 'embed', Dict for 'get_context').
        Raises:
            ValueError: If the operation is unknown or parameters are missing.
            ApiError: For errors during the underlying GraphRAG service calls.
        """
        try:
            if request.operation == 'search':
                return await self._search_graph(request.search_params)
            elif request.operation == 'embed':
                return await self._embed_element(request.embed_params)
            elif request.operation == 'get_context':
                return await self._get_context(request.context_params)
            else:
                raise ValueError(f"Unknown operation: {request.operation}")
        except Exception as e:
            logger.error(f"Error in GraphRagTool._execute: {e}", exc_info=True)
            raise ApiError(f"GraphRAG operation failed: {e}", provider_id=self.provider_id, endpoint=request.operation)

    @trace_async_function(span_name="graph_rag_tool.search_graph")
    async def _search_graph(self, params: GraphRagSearchRequest) -> SearchResults:
        """
        Performs a semantic search on the graph data.
        Args:
            params: GraphRagSearchRequest containing search parameters.
        Returns:
            SearchResults object.
        """
        logger.info(f"Performing graph RAG search for query: '{params.query}'")
        try:
            # Check cache first
            cache_key = f"graph_rag:search:{hash(params.json())}"
            cached_results = self.redis_client.get(
                key=cache_key,
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON
            )
            
            if cached_results:
                logger.info(f"Cache hit for query: '{params.query}'")
                return SearchResults(**cached_results)
            
            # Create search query
            search_query = SearchQuery(
                query=params.query,
                element_types=params.element_types,
                limit=params.limit,
                min_similarity=params.min_similarity,
                include_raw_elements=params.include_raw_elements,
                expansion_strategy=params.expansion_strategy,
                reranking_strategy=params.reranking_strategy,
                chain=params.chain,
                filters=params.filters
            )
            
            # Execute search
            results = await self.graph_rag_service.search(search_query)
            
            # Cache results
            self.redis_client.set(
                key=cache_key,
                value=results.dict(),
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON,
                ttl_seconds=300  # Cache for 5 minutes
            )
            
            logger.info(f"Graph RAG search completed. Found {len(results.results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error during graph RAG search: {e}", exc_info=True)
            raise ApiError(f"Graph RAG search failed: {e}", provider_id=self.provider_id, endpoint="search")

    @trace_async_function(span_name="graph_rag_tool.embed_element")
    async def _embed_element(self, params: GraphRagEmbedRequest) -> Optional[VectorEmbedding]:
        """
        Embeds a specific graph element.
        Args:
            params: GraphRagEmbedRequest containing embedding parameters.
        Returns:
            VectorEmbedding object or None if element not found.
        """
        logger.info(f"Embedding graph element: {params.element_type.value} with ID: {params.element_id}")
        try:
            if params.element_type == GraphElementType.NODE:
                embedding = await self.graph_rag_service.embed_node(
                    node_id=params.element_id,
                    strategy=params.strategy,
                    chain=params.chain
                )
            elif params.element_type == GraphElementType.RELATIONSHIP:
                embedding = await self.graph_rag_service.embed_relationship(
                    relationship_id=params.element_id,
                    strategy=params.strategy
                )
            elif params.element_type == GraphElementType.PATH:
                if not params.start_node_id or not params.end_node_id:
                    raise ValueError("start_node_id and end_node_id are required for path embedding.")
                embedding = await self.graph_rag_service.embed_path(
                    start_node_id=params.start_node_id,
                    end_node_id=params.end_node_id,
                    max_depth=params.depth or 3,  # Default depth if not provided
                    strategy=params.strategy
                )
            elif params.element_type == GraphElementType.SUBGRAPH:
                embedding = await self.graph_rag_service.embed_subgraph(
                    center_node_id=params.element_id,
                    depth=params.depth or 1,  # Default depth if not provided
                    strategy=params.strategy
                )
            else:
                raise ValueError(f"Unsupported element type for embedding: {params.element_type.value}")

            if embedding:
                logger.info(f"Successfully embedded {params.element_type.value} {params.element_id}.")
            else:
                logger.warning(f"Failed to embed {params.element_type.value} {params.element_id}.")
            return embedding
        except Exception as e:
            logger.error(f"Error during graph RAG embedding for {params.element_type.value} {params.element_id}: {e}", exc_info=True)
            raise ApiError(f"Graph RAG embedding failed: {e}", provider_id=self.provider_id, endpoint="embed")

    @trace_async_function(span_name="graph_rag_tool.get_context")
    async def _get_context(self, params: GraphRagContextRequest) -> Dict[str, Any]:
        """
        Retrieves context for a specific topic using graph data.
        Args:
            params: GraphRagContextRequest containing context parameters.
        Returns:
            Dictionary containing context information.
        """
        logger.info(f"Getting context for topic: '{params.topic}'")
        try:
            # Parse topic to determine search strategy
            topic_parts = params.topic.split(':')
            topic_type = topic_parts[0] if len(topic_parts) > 1 else "general"
            topic_value = topic_parts[1] if len(topic_parts) > 1 else params.topic
            
            # Check cache first
            cache_key = f"graph_rag:context:{params.topic}:{params.max_results}:{params.chain or 'all'}"
            cached_context = self.redis_client.get(
                key=cache_key,
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON
            )
            
            if cached_context:
                logger.info(f"Cache hit for context topic: '{params.topic}'")
                return cached_context
            
            # Determine search parameters based on topic type
            if topic_type == "address":
                # Search for address-related information
                search_query = SearchQuery(
                    query=f"Information about blockchain address {topic_value}",
                    element_types=[GraphElementType.NODE, GraphElementType.SUBGRAPH],
                    limit=params.max_results,
                    min_similarity=0.6,
                    include_raw_elements=params.include_raw_data,
                    expansion_strategy=QueryExpansionStrategy.DOMAIN_KNOWLEDGE,
                    chain=params.chain,
                    filters={"labels": ["Address"]}
                )
            elif topic_type == "transaction":
                # Search for transaction-related information
                search_query = SearchQuery(
                    query=f"Information about blockchain transaction {topic_value}",
                    element_types=[GraphElementType.RELATIONSHIP],
                    limit=params.max_results,
                    min_similarity=0.6,
                    include_raw_elements=params.include_raw_data,
                    expansion_strategy=QueryExpansionStrategy.DOMAIN_KNOWLEDGE,
                    chain=params.chain,
                    filters={"type": "TRANSFERRED"}
                )
            else:
                # General topic search
                search_query = SearchQuery(
                    query=params.topic,
                    element_types=[GraphElementType.NODE, GraphElementType.RELATIONSHIP, GraphElementType.SUBGRAPH],
                    limit=params.max_results,
                    min_similarity=0.7,
                    include_raw_elements=params.include_raw_data,
                    expansion_strategy=QueryExpansionStrategy.SEMANTIC,
                    chain=params.chain
                )
            
            # Execute search
            results = await self.graph_rag_service.search(search_query)
            
            # Format context
            context = {
                "topic": params.topic,
                "context_items": [],
                "sources": []
            }
            
            # Extract context items
            for idx, result in enumerate(results.results):
                context_item = {
                    "content": result.text,
                    "relevance": result.similarity,
                    "type": result.element_type.value
                }
                
                if params.include_raw_data and result.raw_element:
                    context_item["raw_data"] = result.raw_element
                
                context["context_items"].append(context_item)
                context["sources"].append(f"{result.element_type.value}:{result.element_id}")
            
            # Add summary if available
            if hasattr(self.graph_rag_service, "summarize") and callable(getattr(self.graph_rag_service, "summarize")):
                try:
                    summary = await self.graph_rag_service.summarize([r.text for r in results.results])
                    context["summary"] = summary
                except Exception as e:
                    logger.warning(f"Failed to generate summary: {e}")
            
            # Cache context
            self.redis_client.set(
                key=cache_key,
                value=context,
                db=RedisDb.CACHE,
                format=SerializationFormat.JSON,
                ttl_seconds=300  # Cache for 5 minutes
            )
            
            logger.info(f"Context retrieval completed for topic: '{params.topic}'. Found {len(context['context_items'])} items.")
            return context
        except Exception as e:
            logger.error(f"Error during context retrieval for topic '{params.topic}': {e}", exc_info=True)
            raise ApiError(f"Context retrieval failed: {e}", provider_id=self.provider_id, endpoint="get_context")

    async def enrich_agent_response(self, query: str, response: str, chain: Optional[ChainType] = None) -> Dict[str, Any]:
        """
        Enriches an agent's response with relevant graph context.
        Args:
            query: The original query that prompted the response.
            response: The agent's response to enrich.
            chain: Optional blockchain chain to filter results by.
        Returns:
            Dictionary containing the enriched response and context.
        """
        logger.info(f"Enriching agent response for query: '{query}'")
        try:
            # Search for relevant context
            search_query = SearchQuery(
                query=query,
                element_types=[GraphElementType.NODE, GraphElementType.SUBGRAPH],
                limit=5,
                min_similarity=0.6,
                include_raw_elements=False,
                expansion_strategy=QueryExpansionStrategy.SEMANTIC,
                chain=chain
            )
            
            results = await self.graph_rag_service.search(search_query)
            
            # Extract context items
            context_items = []
            for result in results.results:
                context_items.append({
                    "content": result.text,
                    "relevance": result.similarity,
                    "source": f"{result.element_type.value}:{result.element_id}"
                })
            
            # Return enriched response
            return {
                "original_query": query,
                "original_response": response,
                "enriched_response": response,  # In a real implementation, this would be modified based on context
                "context": context_items,
                "has_context": len(context_items) > 0
            }
        except Exception as e:
            logger.error(f"Error enriching agent response: {e}", exc_info=True)
            # Return original response if enrichment fails
            return {
                "original_query": query,
                "original_response": response,
                "enriched_response": response,
                "context": [],
                "has_context": False,
                "error": str(e)
            }
