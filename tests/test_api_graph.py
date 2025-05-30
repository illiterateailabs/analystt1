"""
Tests for the graph API endpoints.

This module contains tests for the /api/v1/graph routes, including
schema retrieval, Cypher query execution, and natural language queries.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import app
from backend.api.v1.graph import router as graph_router
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.gemini_client import GeminiClient
from backend.auth.jwt_handler import JWTHandler


# ---- Fixtures ----

@pytest.fixture
def test_user_data():
    """Fixture for test user data."""
    return {
        "id": "test_user_id",
        "email": "test@example.com",
        "full_name": "Test User",
        "role": "analyst"
    }


@pytest.fixture
def access_token(test_user_data):
    """Fixture for a valid access token."""
    return JWTHandler.create_access_token(
        subject=test_user_data["id"],
        user_data=test_user_data
    )


@pytest.fixture
def auth_headers(access_token):
    """Fixture for authentication headers."""
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def mock_neo4j_client():
    """Fixture for mocked Neo4jClient."""
    with patch("backend.api.v1.graph.Neo4jClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        
        # Mock schema retrieval
        mock_instance.get_schema = AsyncMock(return_value={
            "nodes": [
                {"label": "Person", "properties": ["name", "age", "email"]},
                {"label": "Transaction", "properties": ["amount", "timestamp", "currency"]},
                {"label": "Account", "properties": ["number", "balance", "type"]}
            ],
            "relationships": [
                {"type": "SENDS", "start": "Person", "end": "Transaction", "properties": ["fee"]},
                {"type": "RECEIVES", "start": "Transaction", "end": "Person", "properties": []},
                {"type": "OWNS", "start": "Person", "end": "Account", "properties": ["since"]}
            ]
        })
        
        # Mock query execution
        mock_instance.run_query = AsyncMock(return_value=[
            {"n": {"name": "Alice", "age": 30}},
            {"n": {"name": "Bob", "age": 25}}
        ])
        
        # Mock cypher validation
        mock_instance.validate_cypher = AsyncMock(return_value={"valid": True, "message": "Valid Cypher query"})
        
        yield mock_instance


@pytest.fixture
def mock_gemini_client():
    """Fixture for mocked GeminiClient."""
    with patch("backend.api.v1.graph.GeminiClient", autospec=True) as mock_client:
        mock_instance = mock_client.return_value
        
        # Mock Cypher generation from natural language
        mock_instance.generate_cypher_query = AsyncMock(
            return_value="MATCH (p:Person) WHERE p.age > 25 RETURN p.name, p.age"
        )
        
        yield mock_instance


@pytest.fixture
def test_client(mock_neo4j_client, mock_gemini_client):
    """Fixture for test client with mocked dependencies."""
    # Create a test app with mocked dependencies
    test_app = FastAPI()
    test_app.include_router(graph_router, prefix="/api/v1/graph")
    
    # Add dependencies to app state
    test_app.state.neo4j = mock_neo4j_client
    test_app.state.gemini = mock_gemini_client
    
    # Return test client
    with TestClient(test_app) as client:
        yield client


# ---- Tests for GET /api/v1/graph/schema endpoint ----

def test_get_schema_success(test_client, auth_headers, mock_neo4j_client):
    """Test successfully retrieving the graph schema."""
    # Make request
    response = test_client.get(
        "/api/v1/graph/schema",
        headers=auth_headers
    )
    
    # Check response
    assert response.status_code == 200
    assert "nodes" in response.json()
    assert "relationships" in response.json()
    assert len(response.json()["nodes"]) == 3
    assert len(response.json()["relationships"]) == 3
    
    # Verify nodes content
    nodes = response.json()["nodes"]
    assert any(node["label"] == "Person" for node in nodes)
    assert any(node["label"] == "Transaction" for node in nodes)
    assert any(node["label"] == "Account" for node in nodes)
    
    # Verify relationships content
    relationships = response.json()["relationships"]
    assert any(rel["type"] == "SENDS" for rel in relationships)
    assert any(rel["type"] == "RECEIVES" for rel in relationships)
    assert any(rel["type"] == "OWNS" for rel in relationships)
    
    # Verify mock was called correctly
    mock_neo4j_client.get_schema.assert_called_once()


def test_get_schema_no_auth(test_client):
    """Test retrieving the graph schema without authentication."""
    # Make request without auth headers
    response = test_client.get("/api/v1/graph/schema")
    
    # Check response - should require authentication
    assert response.status_code == 401  # Unauthorized


def test_get_schema_error(test_client, auth_headers, mock_neo4j_client):
    """Test retrieving the graph schema when an error occurs."""
    # Mock client to raise exception
    mock_neo4j_client.get_schema.side_effect = Exception("Failed to retrieve schema")
    
    # Make request
    response = test_client.get(
        "/api/v1/graph/schema",
        headers=auth_headers
    )
    
    # Check response
    assert response.status_code == 500  # Internal Server Error
    assert "error" in response.json()
    assert "failed to retrieve schema" in response.json()["error"].lower()


# ---- Tests for POST /api/v1/graph/query endpoint ----

def test_execute_query_success(test_client, auth_headers, mock_neo4j_client):
    """Test successfully executing a Cypher query."""
    # Test data
    test_query = "MATCH (n:Person) RETURN n.name, n.age"
    
    # Make request
    response = test_client.post(
        "/api/v1/graph/query",
        headers=auth_headers,
        json={"query": test_query}
    )
    
    # Check response
    assert response.status_code == 200
    assert "results" in response.json()
    assert len(response.json()["results"]) == 2
    
    # Verify mock was called correctly
    mock_neo4j_client.run_query.assert_called_once_with(test_query, params={})


def test_execute_query_with_params(test_client, auth_headers, mock_neo4j_client):
    """Test executing a Cypher query with parameters."""
    # Test data
    test_query = "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name, n.age"
    test_params = {"min_age": 25}
    
    # Make request
    response = test_client.post(
        "/api/v1/graph/query",
        headers=auth_headers,
        json={"query": test_query, "params": test_params}
    )
    
    # Check response
    assert response.status_code == 200
    assert "results" in response.json()
    
    # Verify mock was called correctly
    mock_neo4j_client.run_query.assert_called_once_with(test_query, params=test_params)


def test_execute_query_empty(test_client, auth_headers):
    """Test executing an empty Cypher query."""
    # Make request with empty query
    response = test_client.post(
        "/api/v1/graph/query",
        headers=auth_headers,
        json={"query": ""}
    )
    
    # Check response
    assert response.status_code == 400  # Bad Request
    assert "empty query" in response.json()["detail"].lower()


def test_execute_query_missing(test_client, auth_headers):
    """Test executing a query with missing query field."""
    # Make request with missing query
    response = test_client.post(
        "/api/v1/graph/query",
        headers=auth_headers,
        json={}
    )
    
    # Check response
    assert response.status_code == 422  # Unprocessable Entity


def test_execute_query_invalid(test_client, auth_headers, mock_neo4j_client):
    """Test executing an invalid Cypher query."""
    # Mock validation to return invalid
    mock_neo4j_client.validate_cypher = AsyncMock(
        return_value={"valid": False, "message": "Invalid Cypher syntax"}
    )
    
    # Mock run_query to raise exception
    mock_neo4j_client.run_query.side_effect = Exception("Invalid Cypher query")
    
    # Test data
    test_query = "MATCH n:Person RETURN n.name"  # Missing parentheses
    
    # Make request
    response = test_client.post(
        "/api/v1/graph/query",
        headers=auth_headers,
        json={"query": test_query}
    )
    
    # Check response
    assert response.status_code == 400  # Bad Request
    assert "error" in response.json()
    assert "invalid cypher" in response.json()["error"].lower()


def test_execute_query_no_auth(test_client):
    """Test executing a query without authentication."""
    # Make request without auth headers
    response = test_client.post(
        "/api/v1/graph/query",
        json={"query": "MATCH (n:Person) RETURN n.name, n.age"}
    )
    
    # Check response - should require authentication
    assert response.status_code == 401  # Unauthorized


def test_execute_query_error(test_client, auth_headers, mock_neo4j_client):
    """Test executing a query when Neo4j client raises an exception."""
    # Mock client to raise exception
    mock_neo4j_client.run_query.side_effect = Exception("Database connection error")
    
    # Make request
    response = test_client.post(
        "/api/v1/graph/query",
        headers=auth_headers,
        json={"query": "MATCH (n:Person) RETURN n.name, n.age"}
    )
    
    # Check response
    assert response.status_code == 500  # Internal Server Error
    assert "error" in response.json()
    assert "database connection error" in response.json()["error"].lower()


# ---- Tests for POST /api/v1/graph/nlq endpoint ----

def test_natural_language_query_success(test_client, auth_headers, mock_neo4j_client, mock_gemini_client):
    """Test successfully executing a natural language query."""
    # Test data
    test_nlq = "Find all people older than 25"
    
    # Make request
    response = test_client.post(
        "/api/v1/graph/nlq",
        headers=auth_headers,
        json={"query": test_nlq}
    )
    
    # Check response
    assert response.status_code == 200
    assert "results" in response.json()
    assert "cypher" in response.json()
    assert "MATCH (p:Person) WHERE p.age > 25" in response.json()["cypher"]
    
    # Verify mocks were called correctly
    mock_gemini_client.generate_cypher_query.assert_called_once_with(test_nlq, schema=mock_neo4j_client.get_schema.return_value)
    mock_neo4j_client.run_query.assert_called_once()


def test_natural_language_query_empty(test_client, auth_headers):
    """Test executing an empty natural language query."""
    # Make request with empty query
    response = test_client.post(
        "/api/v1/graph/nlq",
        headers=auth_headers,
        json={"query": ""}
    )
    
    # Check response
    assert response.status_code == 400  # Bad Request
    assert "empty query" in response.json()["detail"].lower()


def test_natural_language_query_missing(test_client, auth_headers):
    """Test executing a natural language query with missing query field."""
    # Make request with missing query
    response = test_client.post(
        "/api/v1/graph/nlq",
        headers=auth_headers,
        json={}
    )
    
    # Check response
    assert response.status_code == 422  # Unprocessable Entity


def test_natural_language_query_no_auth(test_client):
    """Test executing a natural language query without authentication."""
    # Make request without auth headers
    response = test_client.post(
        "/api/v1/graph/nlq",
        json={"query": "Find all people older than 25"}
    )
    
    # Check response - should require authentication
    assert response.status_code == 401  # Unauthorized


def test_natural_language_query_gemini_error(test_client, auth_headers, mock_gemini_client):
    """Test natural language query when Gemini client raises an exception."""
    # Mock client to raise exception
    mock_gemini_client.generate_cypher_query.side_effect = Exception("Gemini API error")
    
    # Make request
    response = test_client.post(
        "/api/v1/graph/nlq",
        headers=auth_headers,
        json={"query": "Find all people older than 25"}
    )
    
    # Check response
    assert response.status_code == 500  # Internal Server Error
    assert "error" in response.json()
    assert "gemini api error" in response.json()["error"].lower()


def test_natural_language_query_neo4j_error(test_client, auth_headers, mock_neo4j_client, mock_gemini_client):
    """Test natural language query when Neo4j client raises an exception."""
    # Mock Neo4j client to raise exception for query execution
    mock_neo4j_client.run_query.side_effect = Exception("Database query error")
    
    # Make request
    response = test_client.post(
        "/api/v1/graph/nlq",
        headers=auth_headers,
        json={"query": "Find all people older than 25"}
    )
    
    # Check response
    assert response.status_code == 500  # Internal Server Error
    assert "error" in response.json()
    assert "database query error" in response.json()["error"].lower()


def test_natural_language_query_with_context(test_client, auth_headers, mock_neo4j_client, mock_gemini_client):
    """Test natural language query with additional context."""
    # Test data
    test_nlq = "Find all suspicious transactions"
    test_context = "Suspicious transactions are those over $10,000 or with multiple recipients"
    
    # Make request
    response = test_client.post(
        "/api/v1/graph/nlq",
        headers=auth_headers,
        json={"query": test_nlq, "context": test_context}
    )
    
    # Check response
    assert response.status_code == 200
    assert "results" in response.json()
    assert "cypher" in response.json()
    
    # Verify mocks were called correctly with context
    mock_gemini_client.generate_cypher_query.assert_called_once()
    call_args = mock_gemini_client.generate_cypher_query.call_args[0]
    assert call_args[0] == test_nlq
    # Schema should be passed as well
    assert "schema" in mock_gemini_client.generate_cypher_query.call_args[1]


def test_natural_language_query_invalid_cypher(test_client, auth_headers, mock_neo4j_client, mock_gemini_client):
    """Test natural language query that generates invalid Cypher."""
    # Mock Gemini to return invalid Cypher
    mock_gemini_client.generate_cypher_query.return_value = "MATCH n:Person RETURN n"  # Missing parentheses
    
    # Mock Neo4j validation
    mock_neo4j_client.validate_cypher = AsyncMock(
        return_value={"valid": False, "message": "Invalid Cypher syntax"}
    )
    
    # Mock Neo4j query to raise exception
    mock_neo4j_client.run_query.side_effect = Exception("Invalid Cypher query")
    
    # Make request
    response = test_client.post(
        "/api/v1/graph/nlq",
        headers=auth_headers,
        json={"query": "Find all people"}
    )
    
    # Check response - should still return the generated Cypher but with an error
    assert response.status_code == 400
    assert "cypher" in response.json()
    assert "error" in response.json()
    assert "invalid cypher" in response.json()["error"].lower()
