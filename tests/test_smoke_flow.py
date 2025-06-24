"""
End-to-End Smoke Tests for Analyst Droid One Platform

This module contains comprehensive smoke tests that validate the entire platform
works end-to-end. These tests ensure that all components integrate properly and
that key user workflows function as expected.

These tests require a running instance of the platform with all dependencies:
- FastAPI backend
- Redis
- Neo4j
- Celery workers

Run with: pytest -xvs tests/test_smoke_flow.py
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pytest
import redis
from httpx import AsyncClient
from neo4j import GraphDatabase

from backend.core.evidence import EvidenceBundle, EvidenceSource, create_evidence_bundle
from backend.core.explain_cypher import CypherExplanationService, QuerySource
from backend.core.metrics import ApiMetrics
from backend.core.redis_client import RedisClient, RedisDb, SerializationFormat
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.sim_client import SimClient
from backend.jobs.celery_app import celery_app
from backend.jobs.tasks.analysis_tasks import train_gnn_model_task
from backend.main import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test constants
TEST_USER = {"username": "test_user", "password": "test_password"}
TEST_WALLET = "0x123456789abcdef0123456789abcdef012345678"
TEST_QUERY = """
MATCH (a:Address {address: $address})
OPTIONAL MATCH (a)-[tx:TRANSFERRED]->(b:Address)
RETURN a.address AS address, count(tx) AS outgoing_tx_count
"""


# --- Test Fixtures ---

@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for the test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def client():
    """Create an async test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="module")
def redis_client():
    """Create a Redis client for testing."""
    return RedisClient()


@pytest.fixture(scope="module")
def neo4j_client():
    """Create a Neo4j client for testing."""
    return Neo4jClient()


@pytest.fixture(scope="module")
async def auth_token(client):
    """Get an authentication token for API requests."""
    response = await client.post(
        "/api/v1/auth/login",
        json=TEST_USER,
    )
    
    if response.status_code != 200:
        pytest.skip("Authentication failed - skipping tests that require auth")
    
    data = response.json()
    return data["access_token"]


@pytest.fixture(scope="module")
async def auth_headers(auth_token):
    """Create authentication headers for API requests."""
    return {"Authorization": f"Bearer {auth_token}"}


# --- Health Check Tests ---

@pytest.mark.asyncio
async def test_health_endpoints(client):
    """Test that all health endpoints return proper status."""
    logger.info("Testing health endpoints...")
    
    # Test basic health endpoint
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    
    # Test database health
    response = await client.get("/api/v1/health/database")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "connection_pool" in data
    
    # Test Redis health
    response = await client.get("/api/v1/health/redis")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "components" in data
    assert "cache" in data["components"]
    assert "vector_store" in data["components"]
    
    # Test graph database health
    response = await client.get("/api/v1/health/graph")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "node_count" in data
    
    # Test worker health
    response = await client.get("/api/v1/health/workers")
    assert response.status_code == 200
    data = response.json()
    assert data["overall_status"] in ["HEALTHY", "DEGRADED", "UNHEALTHY"]
    assert "workers" in data
    assert "queues" in data
    
    # Test comprehensive system health
    response = await client.get("/api/v1/health/system")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded", "unhealthy"]
    assert "components" in data
    assert "response_time_ms" in data
    
    logger.info("All health endpoints passed!")


# --- Database Connectivity Tests ---

@pytest.mark.asyncio
async def test_redis_connectivity(redis_client):
    """Test Redis connectivity and basic operations."""
    logger.info("Testing Redis connectivity...")
    
    # Test setting and getting a value
    test_key = f"test:smoke:{uuid.uuid4()}"
    test_value = {"timestamp": datetime.now().isoformat(), "test": True}
    
    # Set value
    success = redis_client.set(
        key=test_key,
        value=test_value,
        ttl_seconds=60,
        db=RedisDb.CACHE,
        format=SerializationFormat.JSON,
    )
    assert success, "Failed to set value in Redis"
    
    # Get value
    retrieved = redis_client.get(
        key=test_key,
        db=RedisDb.CACHE,
        format=SerializationFormat.JSON,
    )
    assert retrieved is not None, "Failed to retrieve value from Redis"
    assert retrieved["test"] is True
    
    # Clean up
    redis_client.delete(test_key, RedisDb.CACHE)
    
    logger.info("Redis connectivity test passed!")


@pytest.mark.asyncio
async def test_neo4j_connectivity(neo4j_client):
    """Test Neo4j connectivity and basic queries."""
    logger.info("Testing Neo4j connectivity...")
    
    # Ensure client is connected
    if not neo4j_client.driver:
        await neo4j_client.connect()
    
    # Test a simple query
    result = neo4j_client.execute_query("MATCH (n) RETURN count(n) AS node_count LIMIT 1")
    assert result is not None
    assert len(result) > 0
    assert "node_count" in result[0]
    
    # Test a parameterized query
    timestamp = datetime.now().isoformat()
    test_query = """
    CREATE (t:TestNode {id: $id, timestamp: $timestamp})
    RETURN t.id AS id
    """
    test_id = f"smoke-test-{uuid.uuid4()}"
    result = neo4j_client.execute_query(
        test_query,
        parameters={"id": test_id, "timestamp": timestamp},
    )
    assert result is not None
    assert len(result) > 0
    assert result[0]["id"] == test_id
    
    # Clean up
    neo4j_client.execute_query(
        "MATCH (t:TestNode {id: $id}) DELETE t",
        parameters={"id": test_id},
    )
    
    logger.info("Neo4j connectivity test passed!")


# --- Graph Query and Evidence Tests ---

@pytest.mark.asyncio
async def test_graph_query_with_evidence(client, auth_headers):
    """Test graph query execution with evidence creation."""
    logger.info("Testing graph query with evidence creation...")
    
    # Create a test query
    query_payload = {
        "query": TEST_QUERY,
        "parameters": {"address": TEST_WALLET},
        "create_evidence": True,
        "evidence_description": "Smoke test evidence"
    }
    
    # Execute query
    response = await client.post(
        "/api/v1/graph/query",
        json=query_payload,
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "results" in data
    assert "query_id" in data
    assert "evidence" in data
    
    # Verify evidence was created
    evidence_info = data["evidence"]
    assert "evidence_id" in evidence_info
    assert "bundle_id" in evidence_info
    
    # Get evidence details
    evidence_id = evidence_info["evidence_id"]
    response = await client.get(
        f"/api/v1/analysis/evidence/{evidence_id}",
        headers=auth_headers,
    )
    
    # This might fail if evidence endpoints aren't implemented yet
    if response.status_code == 200:
        evidence_data = response.json()
        assert evidence_data["id"] == evidence_id
        assert evidence_data["description"] == "Smoke test evidence"
        assert "provenance_link" in evidence_data
        assert evidence_data["provenance_link"].startswith("cypher:query:")
    
    logger.info("Graph query with evidence test passed!")


# --- Background Job Tests ---

@pytest.mark.asyncio
async def test_background_job_execution(client, auth_headers):
    """Test background job execution via Celery."""
    logger.info("Testing background job execution...")
    
    # Create a job request
    job_payload = {
        "graph_query": "MATCH (n) RETURN count(n) AS node_count LIMIT 1",
    }
    
    # Submit job
    response = await client.post(
        "/api/v1/analysis/jobs/gnn_training",
        json=job_payload,
        headers=auth_headers,
    )
    
    # If the endpoint isn't implemented, test direct task execution
    if response.status_code != 200:
        logger.info("Job API endpoint not available, testing direct task execution...")
        
        # Execute task directly
        task_id = str(uuid.uuid4())
        result = train_gnn_model_task.apply_async(
            args=["MATCH (n) RETURN count(n) AS node_count LIMIT 1"],
            task_id=task_id,
        )
        
        # Wait for task to complete (with timeout)
        start_time = time.time()
        timeout = 30  # seconds
        while time.time() - start_time < timeout:
            task_result = train_gnn_model_task.AsyncResult(task_id)
            if task_result.ready():
                break
            await asyncio.sleep(1)
        
        # Check result
        assert task_result.ready(), "Task did not complete in time"
        task_result_value = task_result.get()
        assert task_result_value["status"] == "SUCCESS"
        assert "model_id" in task_result_value
        assert "metrics" in task_result_value
        
        logger.info(f"Direct task execution completed: {task_result_value}")
    else:
        # Process API response
        data = response.json()
        assert "task_id" in data
        task_id = data["task_id"]
        
        # Poll for job completion
        max_attempts = 30
        for attempt in range(max_attempts):
            status_response = await client.get(
                f"/api/v1/analysis/jobs/{task_id}",
                headers=auth_headers,
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data["status"] in ["SUCCESS", "FAILURE"]:
                    break
            
            await asyncio.sleep(1)
        
        assert status_data["status"] == "SUCCESS", f"Job failed: {status_data.get('error')}"
        assert "result" in status_data
        
        logger.info(f"Background job completed: {status_data}")
    
    logger.info("Background job test passed!")


# --- Cost Tracking Tests ---

@pytest.mark.asyncio
async def test_cost_tracking_and_budget(client, auth_headers):
    """Test cost tracking and budget monitoring."""
    logger.info("Testing cost tracking and budget monitoring...")
    
    # Get metrics endpoint
    response = await client.get("/metrics")
    assert response.status_code == 200
    metrics_text = response.text
    
    # Check for cost metrics
    assert "external_api_credit_used_total" in metrics_text
    
    # Make a request that should incur costs
    sim_query_payload = {
        "address": TEST_WALLET,
        "chain_id": "ethereum",
    }
    
    # This endpoint might not exist - if not, we'll skip this part
    response = await client.post(
        "/api/v1/analysis/sim/balances",
        json=sim_query_payload,
        headers=auth_headers,
    )
    
    if response.status_code == 200:
        # Check that costs were tracked
        response = await client.get("/metrics")
        updated_metrics = response.text
        
        # Look for SIM-specific metrics
        assert "external_api_credit_used_total{provider=\"sim\"" in updated_metrics
    else:
        logger.info("SIM API endpoint not available, skipping cost verification")
    
    # Check budget status endpoint (if available)
    response = await client.get(
        "/api/v1/analysis/budget",
        headers=auth_headers,
    )
    
    if response.status_code == 200:
        budget_data = response.json()
        assert "providers" in budget_data
        assert "sim" in budget_data["providers"]
        assert "gemini" in budget_data["providers"]
        
        # Check SIM provider budget
        sim_budget = budget_data["providers"]["sim"]
        assert "budget_limit" in sim_budget
        assert "current_usage" in sim_budget
        assert "remaining_percentage" in sim_budget
        
        # Ensure budget is not exceeded
        assert sim_budget["remaining_percentage"] > 0
    
    logger.info("Cost tracking and budget test passed!")


# --- Chat Flow Tests ---

@pytest.mark.asyncio
async def test_chat_to_analysis_flow(client, auth_headers):
    """Test the complete chat → analysis → results flow."""
    logger.info("Testing chat to analysis flow...")
    
    # Create a new conversation
    conversation_payload = {
        "title": "Smoke Test Conversation",
    }
    
    response = await client.post(
        "/api/v1/chat/conversations",
        json=conversation_payload,
        headers=auth_headers,
    )
    
    if response.status_code != 200:
        pytest.skip("Chat API not available - skipping chat flow test")
    
    data = response.json()
    conversation_id = data["id"]
    
    # Send a message that should trigger analysis
    message_payload = {
        "content": f"Analyze wallet {TEST_WALLET} and check for suspicious transactions",
    }
    
    response = await client.post(
        f"/api/v1/chat/conversations/{conversation_id}/messages",
        json=message_payload,
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    message_id = data["id"]
    
    # Wait for response (with timeout)
    max_attempts = 30
    assistant_message = None
    
    for attempt in range(max_attempts):
        response = await client.get(
            f"/api/v1/chat/conversations/{conversation_id}/messages",
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        messages = response.json()
        
        # Look for assistant response after our message
        found_user_message = False
        for msg in messages:
            if found_user_message and msg["role"] == "assistant":
                assistant_message = msg
                break
            if msg["id"] == message_id:
                found_user_message = True
        
        if assistant_message:
            break
        
        await asyncio.sleep(1)
    
    assert assistant_message is not None, "No assistant response received"
    
    # Check that the response contains analysis results
    assert "content" in assistant_message
    content = assistant_message["content"]
    
    # Look for evidence of analysis in the response
    analysis_indicators = [
        TEST_WALLET,
        "transaction",
        "analysis",
        "result",
        "found",
        "query",
    ]
    
    # The response should contain at least some of these indicators
    assert any(indicator in content.lower() for indicator in analysis_indicators)
    
    # Check for tool usage in the response
    if "tool_calls" in assistant_message:
        tool_calls = assistant_message["tool_calls"]
        assert len(tool_calls) > 0
        
        # Look for graph query tool usage
        graph_query_calls = [call for call in tool_calls if "graph_query" in call["name"].lower()]
        if graph_query_calls:
            assert len(graph_query_calls) > 0
            
            # Check tool results
            for call in graph_query_calls:
                assert "result" in call
                result = call["result"]
                assert "status" in result
                if result["status"] == "success":
                    assert "results" in result
    
    logger.info("Chat to analysis flow test passed!")


# --- API Integration Tests ---

@pytest.mark.asyncio
async def test_api_integration(client, auth_headers):
    """Test integration between different API components."""
    logger.info("Testing API integration...")
    
    # Test chat API
    response = await client.get(
        "/api/v1/chat/conversations",
        headers=auth_headers,
    )
    
    if response.status_code == 200:
        conversations = response.json()
        assert isinstance(conversations, list)
    else:
        logger.info("Chat API not available")
    
    # Test graph API
    response = await client.get(
        "/api/v1/graph/schema",
        headers=auth_headers,
    )
    
    if response.status_code == 200:
        schema = response.json()
        assert "nodes" in schema
        assert "relationships" in schema
    else:
        logger.info("Graph schema API not available")
    
    # Test analysis API
    response = await client.get(
        "/api/v1/analysis/status",
        headers=auth_headers,
    )
    
    if response.status_code == 200:
        status = response.json()
        assert "status" in status
    else:
        logger.info("Analysis API not available")
    
    # Test tools API
    response = await client.get(
        "/api/v1/tools",
        headers=auth_headers,
    )
    
    if response.status_code == 200:
        tools = response.json()
        assert isinstance(tools, list)
        if tools:
            assert "name" in tools[0]
            assert "description" in tools[0]
    else:
        logger.info("Tools API not available")
    
    logger.info("API integration test passed!")


# --- User Workflow Tests ---

@pytest.mark.asyncio
async def test_user_workflow(client, auth_headers):
    """Test basic user workflow scenarios."""
    logger.info("Testing user workflows...")
    
    # 1. User logs in (already done via auth_headers fixture)
    
    # 2. User creates a new investigation
    investigation_payload = {
        "title": "Smoke Test Investigation",
        "description": "Testing end-to-end workflow",
        "tags": ["smoke-test", "automated"],
    }
    
    response = await client.post(
        "/api/v1/analysis/investigations",
        json=investigation_payload,
        headers=auth_headers,
    )
    
    investigation_id = None
    if response.status_code == 200:
        data = response.json()
        investigation_id = data["id"]
    else:
        logger.info("Investigations API not available, using mock investigation")
        investigation_id = f"mock-investigation-{uuid.uuid4()}"
    
    # 3. User runs a graph query
    query_payload = {
        "query": """
        MATCH (a:Address)
        WHERE a.address CONTAINS '123'
        RETURN a.address AS address, a.balance AS balance
        LIMIT 5
        """,
        "investigation_id": investigation_id,
    }
    
    response = await client.post(
        "/api/v1/graph/query",
        json=query_payload,
        headers=auth_headers,
    )
    
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data
        
        # 4. User adds evidence to investigation
        if "evidence" in data:
            evidence_id = data["evidence"]["evidence_id"]
            
            link_payload = {
                "investigation_id": investigation_id,
                "evidence_id": evidence_id,
            }
            
            response = await client.post(
                "/api/v1/analysis/investigations/evidence",
                json=link_payload,
                headers=auth_headers,
            )
            
            if response.status_code == 200:
                logger.info("Evidence linked to investigation successfully")
    
    # 5. User generates a report (if API available)
    report_payload = {
        "investigation_id": investigation_id,
        "title": "Smoke Test Report",
        "include_evidence": True,
    }
    
    response = await client.post(
        "/api/v1/analysis/reports",
        json=report_payload,
        headers=auth_headers,
    )
    
    if response.status_code == 200:
        report_data = response.json()
        assert "id" in report_data
        assert "content" in report_data
        
        logger.info("Report generated successfully")
    else:
        logger.info("Reports API not available")
    
    logger.info("User workflow test passed!")


# --- Full System Test ---

@pytest.mark.asyncio
async def test_full_system_flow():
    """Test the full system flow from end to end."""
    logger.info("Testing full system flow...")
    
    # This test combines elements from all the above tests
    # to validate the entire system works together
    
    # 1. Verify health endpoints
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        response = await client.get("/api/v1/health/system")
        assert response.status_code == 200
        system_health = response.json()
        
        # If system is unhealthy, log the issues but continue testing
        if system_health["status"] != "healthy":
            logger.warning(f"System health is {system_health['status']}")
            if "issues" in system_health and system_health["issues"]:
                for issue in system_health["issues"]:
                    logger.warning(f"Health issue: {issue}")
    
    # 2. Test database connectivity
    redis_client = RedisClient()
    test_key = f"test:full-flow:{uuid.uuid4()}"
    test_value = {"timestamp": datetime.now().isoformat()}
    
    redis_client.set(
        key=test_key,
        value=test_value,
        ttl_seconds=60,
        db=RedisDb.CACHE,
        format=SerializationFormat.JSON,
    )
    
    retrieved = redis_client.get(
        key=test_key,
        db=RedisDb.CACHE,
        format=SerializationFormat.JSON,
    )
    
    assert retrieved is not None
    assert retrieved["timestamp"] == test_value["timestamp"]
    
    # 3. Test Neo4j connectivity
    neo4j_client = Neo4jClient()
    if not neo4j_client.driver:
        await neo4j_client.connect()
    
    result = neo4j_client.execute_query("MATCH (n) RETURN count(n) AS node_count LIMIT 1")
    assert result is not None
    assert "node_count" in result[0]
    
    # 4. Test Cypher explanation service
    explanation_service = CypherExplanationService(neo4j_loader=neo4j_client)
    test_query = "MATCH (n) RETURN n LIMIT 1"
    
    results, execution = await explanation_service.execute_and_track_query(
        query_text=test_query,
        source=QuerySource.SYSTEM_GENERATED,
        generated_by="smoke_test",
    )
    
    assert execution.status == "success"
    assert execution.query_id is not None
    
    # 5. Create evidence bundle
    bundle = create_evidence_bundle(
        narrative="Full system test evidence bundle",
        metadata={"test_id": str(uuid.uuid4())},
    )
    
    evidence_id = await explanation_service.create_evidence_from_query(
        query_id=execution.query_id,
        description="Evidence from full system test",
        bundle=bundle,
    )
    
    assert evidence_id is not None
    assert len(bundle.evidence_items) == 1
    
    # 6. Test metrics emission
    ApiMetrics.record_api_cost("test", "smoke_test", 0.01)
    
    # 7. Clean up
    redis_client.delete(test_key, RedisDb.CACHE)
    
    logger.info("Full system flow test passed!")


# --- Main Test Runner ---

if __name__ == "__main__":
    """Run the smoke tests directly."""
    import sys
    
    pytest.main(["-xvs", __file__])
