"""
Tests for SlowAPI rate limiting middleware integration.

This module tests the rate limiting functionality implemented with SlowAPI,
including rate limit headers, exceeded responses, and endpoint-specific limits.
"""

import asyncio
import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address


# Test app with rate limiting
def create_test_app():
    """Create a test FastAPI app with rate limiting."""
    app = FastAPI()
    
    # Configure rate limiter
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    
    # Standard endpoint with default rate limit
    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}
    
    # Endpoint with custom rate limit
    @app.get("/limited", dependencies=[Depends(limiter.limit("2/minute"))])
    async def limited_endpoint():
        return {"message": "rate-limited endpoint"}
    
    # Endpoint with higher rate limit
    @app.get("/high-limit", dependencies=[Depends(limiter.limit("1000/minute"))])
    async def high_limit_endpoint():
        return {"message": "high rate limit endpoint"}
    
    # Endpoint with multiple rate limits
    @app.get("/multi-limit", dependencies=[
        Depends(limiter.limit("5/minute")),
        Depends(limiter.limit("10/hour"))
    ])
    async def multi_limit_endpoint():
        return {"message": "multi-rate-limited endpoint"}
    
    return app


@pytest.fixture
def test_app():
    """Fixture to create a test app with rate limiting."""
    return create_test_app()


@pytest.fixture
def client(test_app):
    """Fixture to create a test client."""
    with TestClient(test_app) as client:
        yield client


def test_rate_limit_headers_present(client):
    """Test that rate limit headers are present in the response."""
    response = client.get("/test")
    assert response.status_code == 200
    
    # Check for rate limit headers
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    assert "X-RateLimit-Reset" in response.headers


def test_custom_rate_limit(client):
    """Test that custom rate limits are applied correctly."""
    # First request should succeed
    response = client.get("/limited")
    assert response.status_code == 200
    
    # Check rate limit headers
    assert response.headers["X-RateLimit-Limit"] == "2"
    assert response.headers["X-RateLimit-Remaining"] == "1"
    
    # Second request should succeed but exhaust the limit
    response = client.get("/limited")
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Remaining"] == "0"
    
    # Third request should be rate limited
    response = client.get("/limited")
    assert response.status_code == 429
    assert "Retry-After" in response.headers


def test_rate_limit_exceeded_response(client):
    """Test the response when rate limit is exceeded."""
    # Make requests until limit is exceeded
    for _ in range(3):  # Assuming limit is 2/minute
        client.get("/limited")
    
    # This request should be rate limited
    response = client.get("/limited")
    assert response.status_code == 429
    assert response.json()["detail"] == "Rate limit exceeded"


def test_different_endpoints_different_limits(client):
    """Test that different endpoints can have different rate limits."""
    # Limited endpoint (2/minute)
    response = client.get("/limited")
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "2"
    
    response = client.get("/limited")
    assert response.status_code == 200
    
    response = client.get("/limited")
    assert response.status_code == 429  # Exceeded
    
    # High limit endpoint (1000/minute)
    response = client.get("/high-limit")
    assert response.status_code == 200
    assert response.headers["X-RateLimit-Limit"] == "1000"
    
    # Should still be able to access high limit endpoint
    for _ in range(5):
        response = client.get("/high-limit")
        assert response.status_code == 200


def test_multi_limit_endpoint(client):
    """Test endpoint with multiple rate limits."""
    # First 5 requests should succeed (5/minute limit)
    for i in range(5):
        response = client.get("/multi-limit")
        assert response.status_code == 200
        assert int(response.headers["X-RateLimit-Remaining"]) == 4 - i
    
    # 6th request should be rate limited
    response = client.get("/multi-limit")
    assert response.status_code == 429


@patch("slowapi.limiter.time.time")
def test_rate_limit_reset(mock_time, client):
    """Test that rate limits reset after the specified time."""
    # Set initial time
    current_time = 1000
    mock_time.return_value = current_time
    
    # Make requests until limit is reached
    response = client.get("/limited")
    assert response.status_code == 200
    
    response = client.get("/limited")
    assert response.status_code == 200
    
    response = client.get("/limited")
    assert response.status_code == 429
    
    # Advance time by 60 seconds (1 minute)
    mock_time.return_value = current_time + 60
    
    # Should be able to make requests again
    response = client.get("/limited")
    assert response.status_code == 200


def test_rate_limit_with_different_clients():
    """Test that rate limits are applied per client."""
    app = create_test_app()
    
    # Create two test clients with different IP addresses
    with TestClient(app, headers={"X-Forwarded-For": "1.2.3.4"}) as client1, \
         TestClient(app, headers={"X-Forwarded-For": "5.6.7.8"}) as client2:
        
        # Client 1 makes requests until limit
        client1.get("/limited")
        client1.get("/limited")
        response = client1.get("/limited")
        assert response.status_code == 429
        
        # Client 2 should still be able to make requests
        response = client2.get("/limited")
        assert response.status_code == 200


def test_rate_limit_in_main_app():
    """
    Integration test with the main application.
    
    This test verifies that rate limiting is correctly applied in the main app.
    """
    from backend.main import app
    
    with TestClient(app) as client:
        # Check that rate limit headers are present
        response = client.get("/health")
        assert response.status_code == 200
        
        # Rate limit headers should be present
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        
        # The actual values will depend on the app configuration
        assert int(response.headers["X-RateLimit-Remaining"]) > 0
