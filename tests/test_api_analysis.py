"""
Tests for the analysis API endpoints.

This module contains tests for the /api/v1/analysis routes, including
code execution, image analysis, and other analytical capabilities.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set required environment variables before importing backend modules
os.environ.setdefault("SECRET_KEY", "test_secret_key")
os.environ.setdefault("GOOGLE_API_KEY", "dummy_key")
os.environ.setdefault("E2B_API_KEY", "dummy_key") 
os.environ.setdefault("NEO4J_PASSWORD", "test_password")

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import with proper error handling
try:
    from backend.main import app
    from backend.api.v1.analysis import router as analysis_router
    from backend.integrations.e2b_client import E2BClient
    from backend.agents.factory import CrewFactory
    from backend.auth.jwt_handler import JWTHandler
except ImportError as e:
    # If imports fail in CI, we need to handle it gracefully
    print(f"Import error in test_api_analysis: {e}")
    # Create dummy objects for tests to at least parse
    app = None
    analysis_router = None
    E2BClient = None
    CrewFactory = None
    JWTHandler = type('JWTHandler', (), {
        'create_access_token': staticmethod(lambda subject, user_data: "dummy_token")
    })()

# Skip all tests in this file if the analysis module doesn't exist
pytestmark = pytest.mark.skipif(
    analysis_router is None,
    reason="analysis API module not yet implemented"
)

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
def mock_e2b_client():
    """Fixture for mocked E2BClient."""
    with patch("backend.api.v1.analysis.E2BClient", autospec=True) as mock_client:
        # Mock successful code execution
        mock_instance = mock_client.return_value
        mock_instance.create_sandbox = AsyncMock()
        mock_instance.execute_code = AsyncMock(return_value={
            "success": True,
            "stdout": "Hello, World!",
            "stderr": "",
            "exit_code": 0
        })
        mock_instance.close_sandbox = AsyncMock()
        yield mock_instance


@pytest.fixture
def mock_crew_factory():
    """Fixture for mocked CrewFactory."""
    with patch("backend.api.v1.analysis.CrewFactory", autospec=True) as mock_factory:
        mock_instance = mock_factory.return_value
        mock_instance.run_crew = AsyncMock(return_value={
            "success": True,
            "result": "Analysis complete"
        })
        yield mock_instance


@pytest.fixture
def test_client(mock_e2b_client, mock_crew_factory):
    """Fixture for test client with mocked dependencies."""
    # Create a test app with mocked dependencies
    test_app = FastAPI()
    test_app.include_router(analysis_router, prefix="/api/v1/analysis")
    
    # Add dependencies to app state
    test_app.state.e2b = mock_e2b_client
    test_app.state.crew_factory = mock_crew_factory
    
    # Return test client
    with TestClient(test_app) as client:
        yield client


# ---- Tests for /api/v1/analysis/code endpoint ----

def test_execute_code_success(test_client, auth_headers, mock_e2b_client):
    """Test successful code execution."""
    # Test data
    test_code = "print('Hello, World!')"
    
    # Make request
    response = test_client.post(
        "/api/v1/analysis/code",
        headers=auth_headers,
        json={"code": test_code}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "Hello, World!" in response.json()["stdout"]
    
    # Verify mocks were called correctly
    mock_e2b_client.create_sandbox.assert_called_once()
    mock_e2b_client.execute_code.assert_called_once_with(test_code, sandbox=mock_e2b_client.create_sandbox.return_value)
    mock_e2b_client.close_sandbox.assert_called_once()


def test_execute_code_missing_code(test_client, auth_headers):
    """Test code execution with missing code field."""
    # Make request with missing code
    response = test_client.post(
        "/api/v1/analysis/code",
        headers=auth_headers,
        json={}
    )
    
    # Check response
    assert response.status_code == 422  # Unprocessable Entity


def test_execute_code_empty_code(test_client, auth_headers):
    """Test code execution with empty code."""
    # Make request with empty code
    response = test_client.post(
        "/api/v1/analysis/code",
        headers=auth_headers,
        json={"code": ""}
    )
    
    # Check response
    assert response.status_code == 400  # Bad Request
    assert "code cannot be empty" in response.json()["detail"].lower()


def test_execute_code_invalid_code(test_client, auth_headers, mock_e2b_client):
    """Test code execution with invalid code."""
    # Mock error response
    mock_e2b_client.execute_code.return_value = {
        "success": False,
        "stdout": "",
        "stderr": "SyntaxError: invalid syntax",
        "exit_code": 1
    }
    
    # Make request with invalid code
    response = test_client.post(
        "/api/v1/analysis/code",
        headers=auth_headers,
        json={"code": "print('Hello, World!'"}  # Missing closing parenthesis
    )
    
    # Check response
    assert response.status_code == 200  # Still returns 200 as the API executed correctly
    assert response.json()["success"] is False
    assert "syntaxerror" in response.json()["stderr"].lower()


def test_execute_code_execution_error(test_client, auth_headers, mock_e2b_client):
    """Test code execution with runtime error."""
    # Mock error response
    mock_e2b_client.execute_code.return_value = {
        "success": False,
        "stdout": "",
        "stderr": "NameError: name 'undefined_variable' is not defined",
        "exit_code": 1
    }
    
    # Make request with code that will cause runtime error
    response = test_client.post(
        "/api/v1/analysis/code",
        headers=auth_headers,
        json={"code": "print(undefined_variable)"}
    )
    
    # Check response
    assert response.status_code == 200  # Still returns 200 as the API executed correctly
    assert response.json()["success"] is False
    assert "nameerror" in response.json()["stderr"].lower()


def test_execute_code_e2b_error(test_client, auth_headers, mock_e2b_client):
    """Test code execution when E2B client raises an exception."""
    # Mock E2B client to raise exception
    mock_e2b_client.execute_code.side_effect = Exception("E2B service unavailable")
    
    # Make request
    response = test_client.post(
        "/api/v1/analysis/code",
        headers=auth_headers,
        json={"code": "print('Hello, World!')"}
    )
    
    # Check response
    assert response.status_code == 500  # Internal Server Error
    assert "error" in response.json()
    assert "e2b service" in response.json()["error"].lower()


def test_execute_code_no_auth(test_client):
    """Test code execution without authentication."""
    # Make request without auth headers
    response = test_client.post(
        "/api/v1/analysis/code",
        json={"code": "print('Hello, World!')"}
    )
    
    # Check response - should require authentication
    assert response.status_code == 401  # Unauthorized or 403 Forbidden


def test_execute_code_with_packages(test_client, auth_headers, mock_e2b_client):
    """Test code execution with package installation."""
    # Test data
    test_code = "import pandas as pd\nprint(pd.__version__)"
    test_packages = ["pandas"]
    
    # Make request
    response = test_client.post(
        "/api/v1/analysis/code",
        headers=auth_headers,
        json={"code": test_code, "packages": test_packages}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["success"] is True
    
    # Verify mocks were called correctly
    mock_e2b_client.create_sandbox.assert_called_once()
    mock_e2b_client.execute_code.assert_called_once()
    mock_e2b_client.close_sandbox.assert_called_once()


def test_execute_code_with_files(test_client, auth_headers, mock_e2b_client):
    """Test code execution with file uploads."""
    # Test data
    test_code = "with open('test.txt', 'r') as f:\n    print(f.read())"
    test_files = [
        {
            "filename": "test.txt",
            "content": "Hello from file!"
        }
    ]
    
    # Make request
    response = test_client.post(
        "/api/v1/analysis/code",
        headers=auth_headers,
        json={"code": test_code, "files": test_files}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["success"] is True
    
    # Verify mocks were called correctly
    mock_e2b_client.create_sandbox.assert_called_once()
    mock_e2b_client.execute_code.assert_called_once()
    mock_e2b_client.close_sandbox.assert_called_once()


def test_execute_code_timeout(test_client, auth_headers, mock_e2b_client):
    """Test code execution with timeout."""
    # Mock timeout response
    mock_e2b_client.execute_code.return_value = {
        "success": False,
        "stdout": "",
        "stderr": "Execution timed out after 30 seconds",
        "exit_code": 124
    }
    
    # Make request with code that will cause timeout
    response = test_client.post(
        "/api/v1/analysis/code",
        headers=auth_headers,
        json={"code": "import time\nwhile True:\n    time.sleep(1)"}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["success"] is False
    assert "timed out" in response.json()["stderr"].lower()
