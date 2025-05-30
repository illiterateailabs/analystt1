"""
Tests for the crew API endpoints.

This module contains tests for the /api/v1/crew routes, including
listing available crews and running specific crews with inputs.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import app
from backend.api.v1.crew import router as crew_router
from backend.agents.factory import CrewFactory
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
def mock_crew_factory():
    """Fixture for mocked CrewFactory."""
    with patch("backend.api.v1.crew.CrewFactory", autospec=True) as mock_factory:
        mock_instance = mock_factory.return_value
        
        # Mock get_available_crews
        mock_instance.get_available_crews = MagicMock(return_value=[
            "fraud_investigation",
            "alert_enrichment",
            "red_blue_simulation",
            "crypto_investigation"
        ])
        
        # Mock run_crew
        mock_instance.run_crew = AsyncMock(return_value={
            "success": True,
            "result": "Crew analysis complete",
            "task_id": "task_123",
            "agent_id": "fraud_pattern_hunter"
        })
        
        # Mock close
        mock_instance.close = AsyncMock()
        
        yield mock_instance


@pytest.fixture
def test_client(mock_crew_factory):
    """Fixture for test client with mocked dependencies."""
    # Create a test app with mocked dependencies
    test_app = FastAPI()
    test_app.include_router(crew_router, prefix="/api/v1/crew")
    
    # Add dependencies to app state
    test_app.state.crew_factory = mock_crew_factory
    
    # Return test client
    with TestClient(test_app) as client:
        yield client


# ---- Tests for GET /api/v1/crew endpoint ----

def test_list_crews_success(test_client, auth_headers, mock_crew_factory):
    """Test successfully listing available crews."""
    # Make request
    response = test_client.get(
        "/api/v1/crew",
        headers=auth_headers
    )
    
    # Check response
    assert response.status_code == 200
    assert "crews" in response.json()
    assert len(response.json()["crews"]) == 4
    assert "fraud_investigation" in response.json()["crews"]
    assert "crypto_investigation" in response.json()["crews"]
    
    # Verify mock was called correctly
    mock_crew_factory.get_available_crews.assert_called_once()


def test_list_crews_no_auth(test_client):
    """Test listing crews without authentication."""
    # Make request without auth headers
    response = test_client.get("/api/v1/crew")
    
    # Check response - should require authentication
    assert response.status_code == 401  # Unauthorized


def test_list_crews_error(test_client, auth_headers, mock_crew_factory):
    """Test listing crews when an error occurs."""
    # Mock factory to raise exception
    mock_crew_factory.get_available_crews.side_effect = Exception("Failed to get crews")
    
    # Make request
    response = test_client.get(
        "/api/v1/crew",
        headers=auth_headers
    )
    
    # Check response
    assert response.status_code == 500  # Internal Server Error
    assert "error" in response.json()
    assert "failed to get crews" in response.json()["error"].lower()


# ---- Tests for POST /api/v1/crew/{crew_name} endpoint ----

def test_run_crew_success(test_client, auth_headers, mock_crew_factory):
    """Test successfully running a crew."""
    # Test data
    crew_name = "fraud_investigation"
    inputs = {
        "query": "Identify suspicious transactions over $10,000",
        "time_range": "last 30 days"
    }
    
    # Make request
    response = test_client.post(
        f"/api/v1/crew/{crew_name}",
        headers=auth_headers,
        json={"inputs": inputs}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "result" in response.json()
    assert "crew analysis complete" in response.json()["result"].lower()
    
    # Verify mock was called correctly
    mock_crew_factory.run_crew.assert_called_once_with(crew_name, inputs=inputs)
    mock_crew_factory.close.assert_called_once()


def test_run_crew_invalid_name(test_client, auth_headers, mock_crew_factory):
    """Test running a crew with an invalid name."""
    # Mock get_available_crews to return valid crew names
    mock_crew_factory.get_available_crews.return_value = [
        "fraud_investigation", "alert_enrichment"
    ]
    
    # Make request with invalid crew name
    response = test_client.post(
        "/api/v1/crew/nonexistent_crew",
        headers=auth_headers,
        json={"inputs": {}}
    )
    
    # Check response
    assert response.status_code == 404  # Not Found
    assert "error" in response.json()
    assert "crew not found" in response.json()["error"].lower()
    
    # Verify mock was called correctly
    mock_crew_factory.get_available_crews.assert_called_once()
    # run_crew should not be called
    mock_crew_factory.run_crew.assert_not_called()


def test_run_crew_no_inputs(test_client, auth_headers, mock_crew_factory):
    """Test running a crew without inputs."""
    # Test data
    crew_name = "fraud_investigation"
    
    # Make request without inputs
    response = test_client.post(
        f"/api/v1/crew/{crew_name}",
        headers=auth_headers,
        json={}
    )
    
    # Check response - should still work with default empty inputs
    assert response.status_code == 200
    assert response.json()["success"] is True
    
    # Verify mock was called correctly with empty inputs
    mock_crew_factory.run_crew.assert_called_once_with(crew_name, inputs={})


def test_run_crew_empty_inputs(test_client, auth_headers, mock_crew_factory):
    """Test running a crew with empty inputs."""
    # Test data
    crew_name = "fraud_investigation"
    
    # Make request with empty inputs
    response = test_client.post(
        f"/api/v1/crew/{crew_name}",
        headers=auth_headers,
        json={"inputs": {}}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["success"] is True
    
    # Verify mock was called correctly with empty inputs
    mock_crew_factory.run_crew.assert_called_once_with(crew_name, inputs={})


def test_run_crew_complex_inputs(test_client, auth_headers, mock_crew_factory):
    """Test running a crew with complex nested inputs."""
    # Test data
    crew_name = "crypto_investigation"
    inputs = {
        "address": "0x1234567890abcdef1234567890abcdef12345678",
        "blockchain": "ethereum",
        "time_range": {
            "start": "2023-01-01T00:00:00Z",
            "end": "2023-12-31T23:59:59Z"
        },
        "filters": {
            "min_value": 1.0,
            "max_value": 1000.0,
            "transaction_types": ["transfer", "swap", "deposit"]
        },
        "options": {
            "include_metadata": True,
            "trace_hops": 3,
            "cluster_addresses": True
        }
    }
    
    # Make request
    response = test_client.post(
        f"/api/v1/crew/{crew_name}",
        headers=auth_headers,
        json={"inputs": inputs}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["success"] is True
    
    # Verify mock was called correctly with complex inputs
    mock_crew_factory.run_crew.assert_called_once_with(crew_name, inputs=inputs)


def test_run_crew_no_auth(test_client):
    """Test running a crew without authentication."""
    # Make request without auth headers
    response = test_client.post(
        "/api/v1/crew/fraud_investigation",
        json={"inputs": {}}
    )
    
    # Check response - should require authentication
    assert response.status_code == 401  # Unauthorized


def test_run_crew_error(test_client, auth_headers, mock_crew_factory):
    """Test running a crew when an error occurs."""
    # Mock factory to raise exception
    mock_crew_factory.run_crew.side_effect = Exception("Failed to run crew")
    
    # Make request
    response = test_client.post(
        "/api/v1/crew/fraud_investigation",
        headers=auth_headers,
        json={"inputs": {}}
    )
    
    # Check response
    assert response.status_code == 500  # Internal Server Error
    assert "error" in response.json()
    assert "failed to run crew" in response.json()["error"].lower()


def test_run_crew_execution_failure(test_client, auth_headers, mock_crew_factory):
    """Test running a crew that fails during execution."""
    # Mock run_crew to return failure
    mock_crew_factory.run_crew.return_value = {
        "success": False,
        "error": "Agent execution failed: insufficient data"
    }
    
    # Make request
    response = test_client.post(
        "/api/v1/crew/fraud_investigation",
        headers=auth_headers,
        json={"inputs": {}}
    )
    
    # Check response
    assert response.status_code == 200  # Still returns 200 as the API executed correctly
    assert response.json()["success"] is False
    assert "error" in response.json()
    assert "insufficient data" in response.json()["error"].lower()
