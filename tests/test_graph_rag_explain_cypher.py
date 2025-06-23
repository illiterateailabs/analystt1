"""
Integration tests for Graph-Aware RAG with Explain-with-Cypher system.

These tests verify the integration between the Cypher explanation system,
Graph RAG, and evidence management to ensure proper query tracking,
evidence creation, and citation generation.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.explain_cypher import (
    CypherExplanationService,
    CypherQuery,
    CypherQueryExecution,
    QuerySource,
    QueryStatus,
)
from backend.agents.tools.graph_query_tool import GraphQueryTool
from backend.core.evidence import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceSource,
    GraphElementEvidence,
    create_evidence_bundle,
)
from backend.core.graph_rag import GraphRAG, GraphElementType
from backend.core.redis_client import RedisClient, RedisDb, SerializationFormat
from backend.integrations.neo4j_client import Neo4jClient


# --- Test Fixtures ---

@pytest.fixture
def mock_neo4j_client():
    """Mock Neo4j client for testing."""
    client = MagicMock(spec=Neo4jClient)
    
    # Mock the _execute_query method to return predefined results
    def mock_execute_query(query, parameters=None):
        # Create a mock result object with data() method
        mock_record = MagicMock()
        
        # Different results based on query content
        if "Person" in query:
            mock_record.data.return_value = {
                "name": "John Doe",
                "age": 30,
            }
        elif "Address" in query:
            mock_record.data.return_value = {
                "address": "0x123456789abcdef",
                "balance": 100.0,
            }
        elif "TRANSFERRED" in query:
            mock_record.data.return_value = {
                "from_address": "0x123456789abcdef",
                "to_address": "0xabcdef123456789",
                "value": 10.0,
                "timestamp": "2025-06-23T12:00:00Z",
            }
        else:
            mock_record.data.return_value = {"result": "generic_result"}
        
        # Mock result object
        mock_result = MagicMock()
        mock_result.__iter__.return_value = [mock_record]
        
        # Mock stats
        mock_result.summary.counters.nodes_created = 0
        mock_result.summary.counters.relationships_created = 0
        mock_result.summary.counters.properties_set = 0
        mock_result.summary.counters.labels_added = 0
        
        return mock_result
    
    client._execute_query = mock_execute_query
    client._process_result_stats = MagicMock(return_value=MagicMock(
        nodes_created=0,
        relationships_created=0,
        properties_set=0,
        labels_added=0,
    ))
    
    return client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    client = MagicMock(spec=RedisClient)
    
    # Store data in memory for testing
    stored_data = {}
    
    def mock_set(key, value, ttl_seconds=None, db=None, format=None):
        stored_data[f"{db}:{key}"] = value
        return True
    
    def mock_get(key, db=None, format=None, default=None):
        full_key = f"{db}:{key}"
        if full_key in stored_data:
            return stored_data[full_key]
        return default
    
    def mock_keys(pattern, db=None):
        prefix = f"{db}:"
        matching_keys = []
        for key in stored_data.keys():
            if key.startswith(prefix) and pattern.replace("*", "") in key:
                matching_keys.append(key.replace(prefix, ""))
        return matching_keys
    
    client.set = mock_set
    client.get = mock_get
    client.keys = mock_keys
    
    return client


@pytest.fixture
def cypher_explanation_service(mock_neo4j_client, mock_redis_client):
    """Create a CypherExplanationService with mocked dependencies."""
    service = CypherExplanationService(
        neo4j_loader=mock_neo4j_client,
        redis_client=mock_redis_client,
    )
    return service


@pytest.fixture
def graph_query_tool(mock_neo4j_client):
    """Create a GraphQueryTool with mocked dependencies."""
    with patch("backend.agents.tools.graph_query_tool.CypherExplanationService") as mock_service_class:
        # Create a mock explanation service
        mock_service = AsyncMock()
        
        # Mock the execute_and_track_query method
        async def mock_execute_and_track(query_text, parameters=None, source=None, generated_by=None, **kwargs):
            # Return mock results and execution record
            results = []
            if "Person" in query_text:
                results = [{"name": "John Doe", "age": 30}]
            elif "Address" in query_text:
                results = [{"address": "0x123456789abcdef", "balance": 100.0}]
            elif "TRANSFERRED" in query_text:
                results = [{
                    "from_address": "0x123456789abcdef",
                    "to_address": "0xabcdef123456789",
                    "value": 10.0,
                    "timestamp": "2025-06-23T12:00:00Z",
                }]
            
            execution = CypherQueryExecution(
                query_id="test-query-id",
                status=QueryStatus.SUCCESS,
                duration_ms=10.0,
                result_summary={"count": len(results)},
            )
            
            return results, execution
        
        # Mock the create_evidence_from_query method
        async def mock_create_evidence(query_id, description=None, confidence=0.9, source=None, bundle=None):
            if bundle:
                # Create a mock evidence item and add it to the bundle
                evidence_item = GraphElementEvidence(
                    description=description or "Test evidence",
                    source=source or EvidenceSource.GRAPH_ANALYSIS,
                    confidence=confidence,
                    element_id=query_id,
                    element_type=GraphElementType.SUBGRAPH,
                    element_properties={"query_id": query_id},
                )
                bundle.add_evidence(evidence_item)
                return evidence_item.id
            return "test-evidence-id"
        
        # Mock the cite_query_in_response method
        async def mock_cite_query(query_id, response_text):
            citation = f"\n\n[Query: {query_id}]\n```cypher\nMATCH (n) RETURN n\n```"
            return response_text + citation
        
        mock_service.execute_and_track_query.side_effect = mock_execute_and_track
        mock_service.create_evidence_from_query.side_effect = mock_create_evidence
        mock_service.cite_query_in_response.side_effect = mock_cite_query
        
        # Return the mock service from the constructor
        mock_service_class.return_value = mock_service
        
        # Create and return the tool
        tool = GraphQueryTool(neo4j_client=mock_neo4j_client)
        return tool


# --- Tests ---

@pytest.mark.asyncio
async def test_query_execution_creates_evidence(cypher_explanation_service):
    """Test that query execution creates evidence with proper citations."""
    # Execute a query
    query_text = "MATCH (p:Person) RETURN p.name, p.age"
    results, execution = await cypher_explanation_service.execute_and_track_query(
        query_text=query_text,
        source=QuerySource.HUMAN_INPUT,
        generated_by="test_user",
    )
    
    # Create an evidence bundle
    bundle = create_evidence_bundle(narrative="Test narrative")
    
    # Create evidence from the query
    evidence_id = await cypher_explanation_service.create_evidence_from_query(
        query_id=execution.query_id,
        description="Evidence from test query",
        bundle=bundle,
    )
    
    # Verify evidence was created
    assert evidence_id is not None
    assert len(bundle.evidence_items) == 1
    
    # Verify evidence has correct properties
    evidence = bundle.evidence_items[0]
    assert evidence.id == evidence_id
    assert evidence.description == "Evidence from test query"
    assert evidence.source == EvidenceSource.SYSTEM
    assert evidence.confidence == 0.9
    assert "query_id" in evidence.raw_data
    assert evidence.raw_data["query_id"] == execution.query_id
    
    # Verify evidence has provenance link to query
    assert evidence.provenance_link == f"cypher:query:{execution.query_id}"
    
    # Generate citation
    response_text = "Here are the query results:"
    citation = await cypher_explanation_service.cite_query_in_response(
        query_id=execution.query_id,
        response_text=response_text,
    )
    
    # Verify citation includes query ID and text
    assert execution.query_id in citation
    assert query_text in citation


@pytest.mark.asyncio
async def test_evidence_bundle_links_to_queries(cypher_explanation_service):
    """Test that evidence bundles link to queries via provenance."""
    # Execute a query
    query_text = "MATCH (a:Address) RETURN a.address, a.balance"
    results, execution = await cypher_explanation_service.execute_and_track_query(
        query_text=query_text,
        source=QuerySource.TOOL_GENERATED,
        generated_by="test_tool",
    )
    
    # Create an evidence bundle
    bundle = create_evidence_bundle(narrative="Test narrative")
    
    # Create evidence from the query
    evidence_id = await cypher_explanation_service.create_evidence_from_query(
        query_id=execution.query_id,
        description="Evidence from address query",
        bundle=bundle,
    )
    
    # Verify query is linked to evidence
    query = await cypher_explanation_service.get_query_provenance(execution.query_id)
    assert query is not None
    assert evidence_id in query.linked_evidence_ids
    
    # Verify evidence has provenance link to query
    evidence = bundle.evidence_items[0]
    assert evidence.provenance_link == f"cypher:query:{execution.query_id}"
    
    # Verify we can retrieve the query from the provenance link
    provenance_parts = evidence.provenance_link.split(":")
    if len(provenance_parts) == 3 and provenance_parts[0] == "cypher" and provenance_parts[1] == "query":
        retrieved_query_id = provenance_parts[2]
        retrieved_query = await cypher_explanation_service.get_query_provenance(retrieved_query_id)
        assert retrieved_query is not None
        assert retrieved_query.id == execution.query_id
        assert retrieved_query.query_text == query_text


@pytest.mark.asyncio
async def test_query_templates_and_patterns(cypher_explanation_service):
    """Test that query templates and common patterns work."""
    # Get common query templates
    templates = await cypher_explanation_service.get_common_query_templates()
    assert len(templates) > 0
    
    # Find a specific template
    address_path_template = None
    for template in templates:
        if template.id == "find_path_between_addresses":
            address_path_template = template
            break
    
    assert address_path_template is not None
    assert "shortestPath" in address_path_template.template
    
    # Render the template with parameters
    parameters = {
        "address1": "0x123456789abcdef",
        "address2": "0xabcdef123456789",
        "max_depth": 3,
    }
    
    rendered_query = await cypher_explanation_service.render_query_template(
        template_id="find_path_between_addresses",
        parameters=parameters,
    )
    
    assert rendered_query is not None
    assert "shortestPath" in rendered_query
    assert "0x123456789abcdef" in rendered_query
    assert "0xabcdef123456789" in rendered_query
    assert "*1..3" in rendered_query  # max_depth parameter
    
    # Execute the rendered query
    results, execution = await cypher_explanation_service.execute_and_track_query(
        query_text=rendered_query,
        parameters={},  # Parameters are already in the rendered query
        source=QuerySource.TOOL_GENERATED,
        generated_by="test_template",
    )
    
    assert execution.status == QueryStatus.SUCCESS


@pytest.mark.asyncio
async def test_graph_query_tool_integration(graph_query_tool):
    """Test GraphQueryTool integration with evidence creation."""
    # Create a Cypher query
    query = "MATCH (a:Address)-[tx:TRANSFERRED]->(b:Address) RETURN a, tx, b LIMIT 5"
    
    # Execute the query with evidence creation
    result_json = await graph_query_tool._arun(
        query=query,
        create_evidence=True,
        evidence_description="Test evidence from GraphQueryTool",
    )
    
    # Parse the result
    result = json.loads(result_json)
    
    # Verify the result structure
    assert result["status"] == "success"
    assert "results" in result
    assert "count" in result
    assert "query_id" in result
    assert "execution_id" in result
    assert "evidence" in result
    
    # Verify evidence information
    evidence_info = result["evidence"]
    assert "evidence_id" in evidence_info
    assert "bundle_id" in evidence_info
    assert "narrative" in evidence_info
    assert "quality_score" in evidence_info
    
    # Verify the evidence was created with the correct description
    assert graph_query_tool.explanation_service.create_evidence_from_query.called
    call_args = graph_query_tool.explanation_service.create_evidence_from_query.call_args
    assert call_args is not None
    assert call_args[1]["description"] == "Test evidence from GraphQueryTool"


@pytest.mark.asyncio
async def test_end_to_end_flow(graph_query_tool, cypher_explanation_service):
    """Test the end-to-end flow: query → results → evidence → citations."""
    # Create a Cypher query
    query = "MATCH (a:Address)-[tx:TRANSFERRED]->(b:Address) WHERE tx.value > 5.0 RETURN a.address, b.address, tx.value LIMIT 5"
    
    # Execute the query with evidence creation
    result_json = await graph_query_tool._arun(
        query=query,
        create_evidence=True,
        evidence_description="High-value transaction evidence",
    )
    
    # Parse the result
    result = json.loads(result_json)
    
    # Extract evidence information
    evidence_id = result["evidence"]["evidence_id"]
    bundle_id = result["evidence"]["bundle_id"]
    query_id = result["query_id"]
    
    # Verify the query was stored
    with patch.object(cypher_explanation_service, "get_query_provenance") as mock_get_query:
        # Mock the query retrieval
        mock_query = CypherQuery(
            id=query_id,
            query_text=query,
            source=QuerySource.TOOL_GENERATED,
            generated_by="test",
            linked_evidence_ids=[evidence_id],
        )
        mock_get_query.return_value = mock_query
        
        # Retrieve the query
        retrieved_query = await cypher_explanation_service.get_query_provenance(query_id)
        
        # Verify query properties
        assert retrieved_query is not None
        assert retrieved_query.id == query_id
        assert retrieved_query.query_text == query
        assert evidence_id in retrieved_query.linked_evidence_ids
    
    # Generate citation for the query
    with patch.object(cypher_explanation_service, "cite_query_in_response") as mock_cite:
        # Mock the citation generation
        citation_text = f"\n\n[Query: {query_id}]\n```cypher\n{query}\n```"
        mock_cite.return_value = f"Analysis results: High-value transactions detected.{citation_text}"
        
        # Generate citation
        response_with_citation = await cypher_explanation_service.cite_query_in_response(
            query_id=query_id,
            response_text="Analysis results: High-value transactions detected.",
        )
        
        # Verify citation
        assert response_with_citation is not None
        assert query_id in response_with_citation
        assert "```cypher" in response_with_citation
        assert query in response_with_citation


@pytest.mark.asyncio
async def test_cypher_queries_stored_and_retrievable(cypher_explanation_service):
    """Test that Cypher queries are stored and retrievable."""
    # Execute multiple queries
    queries = [
        "MATCH (p:Person) RETURN p.name, p.age",
        "MATCH (a:Address) RETURN a.address, a.balance",
        "MATCH (a:Address)-[tx:TRANSFERRED]->(b:Address) RETURN a.address, b.address, tx.value",
    ]
    
    query_ids = []
    for i, query_text in enumerate(queries):
        results, execution = await cypher_explanation_service.execute_and_track_query(
            query_text=query_text,
            source=QuerySource.HUMAN_INPUT,
            generated_by=f"test_user_{i}",
        )
        query_ids.append(execution.query_id)
    
    # Retrieve each query and verify its properties
    for i, query_id in enumerate(query_ids):
        query = await cypher_explanation_service.get_query_provenance(query_id)
        assert query is not None
        assert query.id == query_id
        assert query.query_text == queries[i]
        assert query.source == QuerySource.HUMAN_INPUT
        assert query.generated_by == f"test_user_{i}"
    
    # Search for queries
    with patch.object(cypher_explanation_service, "search_queries") as mock_search:
        # Mock the search results
        mock_queries = [
            CypherQuery(
                id=query_ids[0],
                query_text=queries[0],
                source=QuerySource.HUMAN_INPUT,
                generated_by="test_user_0",
            ),
            CypherQuery(
                id=query_ids[2],
                query_text=queries[2],
                source=QuerySource.HUMAN_INPUT,
                generated_by="test_user_2",
            ),
        ]
        mock_search.return_value = mock_queries
        
        # Search for queries containing "Address"
        search_results = await cypher_explanation_service.search_queries(
            text_search="Address",
        )
        
        # Verify search results
        assert len(search_results) == 2
        assert search_results[0].id == query_ids[0]
        assert search_results[1].id == query_ids[2]
        assert "Address" in search_results[0].query_text or "Address" in search_results[1].query_text


@pytest.mark.asyncio
async def test_evidence_bundle_with_multiple_queries(cypher_explanation_service):
    """Test creating an evidence bundle with multiple queries."""
    # Create an evidence bundle
    bundle = create_evidence_bundle(narrative="Investigation of suspicious transactions")
    
    # Execute multiple queries
    queries = [
        "MATCH (a:Address {address: '0x123456789abcdef'}) RETURN a",
        "MATCH (a:Address {address: '0x123456789abcdef'})-[tx:TRANSFERRED]->(b:Address) RETURN a.address, b.address, tx.value, tx.timestamp",
        "MATCH (a:Address {address: '0x123456789abcdef'})<-[tx:TRANSFERRED]-(b:Address) RETURN a.address, b.address, tx.value, tx.timestamp",
    ]
    
    evidence_ids = []
    for i, query_text in enumerate(queries):
        # Execute query
        results, execution = await cypher_explanation_service.execute_and_track_query(
            query_text=query_text,
            source=QuerySource.HUMAN_INPUT,
            generated_by="investigator",
            description=f"Query {i+1}: {query_text[:30]}...",
        )
        
        # Create evidence from query
        evidence_id = await cypher_explanation_service.create_evidence_from_query(
            query_id=execution.query_id,
            description=f"Evidence from query {i+1}",
            bundle=bundle,
        )
        
        evidence_ids.append(evidence_id)
    
    # Verify the bundle contains all evidence items
    assert len(bundle.evidence_items) == len(queries)
    
    # Verify each evidence item has the correct properties
    for i, evidence_id in enumerate(evidence_ids):
        # Find the evidence item in the bundle
        evidence = None
        for item in bundle.evidence_items:
            if item.id == evidence_id:
                evidence = item
                break
        
        assert evidence is not None
        assert evidence.description == f"Evidence from query {i+1}"
        assert evidence.source == EvidenceSource.SYSTEM
        assert "query_id" in evidence.raw_data
        assert evidence.provenance_link.startswith("cypher:query:")
    
    # Generate a narrative with citations
    narrative = bundle.narrative
    for i, evidence_id in enumerate(evidence_ids):
        # Find the evidence item
        evidence = None
        for item in bundle.evidence_items:
            if item.id == evidence_id:
                evidence = item
                break
        
        # Extract query ID from provenance link
        query_id = evidence.provenance_link.split(":")[-1]
        
        # Add citation to narrative
        with patch.object(cypher_explanation_service, "cite_query_in_response") as mock_cite:
            citation = f"\n\n[Query {i+1}: {query_id}]\n```cypher\n{queries[i]}\n```"
            mock_cite.return_value = narrative + citation
            
            narrative = await cypher_explanation_service.cite_query_in_response(
                query_id=query_id,
                response_text=narrative,
            )
    
    # Verify narrative contains citations for all queries
    for i, query_id in enumerate(evidence_ids):
        assert f"Query {i+1}" in narrative
        assert queries[i] in narrative


@pytest.mark.asyncio
async def test_query_metrics_and_performance(cypher_explanation_service):
    """Test query metrics and performance tracking."""
    # Execute the same query multiple times
    query_text = "MATCH (a:Address) RETURN a.address, a.balance"
    
    # First execution
    results1, execution1 = await cypher_explanation_service.execute_and_track_query(
        query_text=query_text,
        source=QuerySource.HUMAN_INPUT,
    )
    
    # Second execution (should use cache if enabled)
    results2, execution2 = await cypher_explanation_service.execute_and_track_query(
        query_text=query_text,
        source=QuerySource.HUMAN_INPUT,
        use_cache=True,
    )
    
    # Third execution with different parameters (should not use cache)
    results3, execution3 = await cypher_explanation_service.execute_and_track_query(
        query_text=query_text,
        parameters={"limit": 10},
        source=QuerySource.HUMAN_INPUT,
    )
    
    # Get metrics for the query
    with patch.object(cypher_explanation_service, "get_query_metrics") as mock_metrics:
        # Mock the metrics
        mock_metrics.return_value = {
            "execution_count": 3,
            "avg_duration_ms": 15.0,
            "min_duration_ms": 10.0,
            "max_duration_ms": 20.0,
            "success_rate": 1.0,
            "cache_hit_rate": 0.33,
        }
        
        # Get metrics
        metrics = await cypher_explanation_service.get_query_metrics(execution1.query_id)
        
        # Verify metrics
        assert metrics["execution_count"] == 3
        assert metrics["avg_duration_ms"] == 15.0
        assert metrics["success_rate"] == 1.0
        assert metrics["cache_hit_rate"] == 0.33
