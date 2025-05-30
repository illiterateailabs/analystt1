"""
Tests for the chat API endpoints.

This module contains tests for the /api/v1/chat routes, including
message handling, conversation history, and streaming responses.
"""

import json
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import app
from backend.api.v1.chat import router as chat_router
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
def mock_gemini_client():
    """Fixture for mocked GeminiClient."""
    with patch("backend.api.v1.chat.GeminiClient", autospec=True) as mock_client:
        # Mock successful text generation
        mock_instance = mock_client.return_value
        mock_instance.generate_text = AsyncMock(return_value="This is a test response from Gemini.")
        
        # Mock streaming response
        async def mock_stream_text(*args, **kwargs):
            yield "This "
            yield "is "
            yield "a "
            yield "test "
            yield "response "
            yield "from "
            yield "Gemini."
        
        mock_instance.stream_text = AsyncMock(side_effect=mock_stream_text)
        yield mock_instance


@pytest.fixture
def test_client(mock_gemini_client):
    """Fixture for test client with mocked dependencies."""
    # Create a test app with mocked dependencies
    test_app = FastAPI()
    test_app.include_router(chat_router, prefix="/api/v1/chat")
    
    # Add dependencies to app state
    test_app.state.gemini = mock_gemini_client
    
    # Return test client
    with TestClient(test_app) as client:
        yield client


# ---- Tests for /api/v1/chat endpoint ----

def test_chat_message_success(test_client, auth_headers, mock_gemini_client):
    """Test successful chat message."""
    # Test data
    test_message = "Hello, how are you?"
    
    # Make request
    response = test_client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": test_message}
    )
    
    # Check response
    assert response.status_code == 200
    assert "response" in response.json()
    assert response.json()["response"] == "This is a test response from Gemini."
    
    # Verify mock was called correctly
    mock_gemini_client.generate_text.assert_called_once_with(test_message, context=None)


def test_chat_message_with_context(test_client, auth_headers, mock_gemini_client):
    """Test chat message with context."""
    # Test data
    test_message = "Tell me more about this"
    test_context = "This is some context about financial fraud detection."
    
    # Make request
    response = test_client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": test_message, "context": test_context}
    )
    
    # Check response
    assert response.status_code == 200
    assert "response" in response.json()
    
    # Verify mock was called correctly
    mock_gemini_client.generate_text.assert_called_once_with(test_message, context=test_context)


def test_chat_message_with_history(test_client, auth_headers, mock_gemini_client):
    """Test chat message with conversation history."""
    # Test data
    test_message = "What else can you tell me?"
    test_history = [
        {"role": "user", "content": "Tell me about fraud detection."},
        {"role": "assistant", "content": "Fraud detection involves identifying suspicious patterns."},
    ]
    
    # Make request
    response = test_client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": test_message, "history": test_history}
    )
    
    # Check response
    assert response.status_code == 200
    assert "response" in response.json()
    
    # Verify mock was called correctly - the history should be formatted into context
    mock_gemini_client.generate_text.assert_called_once()
    # Check that history was included in the call
    call_args = mock_gemini_client.generate_text.call_args[1]
    assert "context" in call_args
    assert "fraud detection" in call_args["context"].lower()


def test_chat_message_empty(test_client, auth_headers):
    """Test chat message with empty message."""
    # Make request with empty message
    response = test_client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": ""}
    )
    
    # Check response
    assert response.status_code == 400  # Bad Request
    assert "empty message" in response.json()["detail"].lower()


def test_chat_message_missing(test_client, auth_headers):
    """Test chat message with missing message field."""
    # Make request with missing message
    response = test_client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={}
    )
    
    # Check response
    assert response.status_code == 422  # Unprocessable Entity


def test_chat_message_no_auth(test_client):
    """Test chat message without authentication."""
    # Make request without auth headers
    response = test_client.post(
        "/api/v1/chat",
        json={"message": "Hello, how are you?"}
    )
    
    # Check response - should require authentication
    assert response.status_code == 401  # Unauthorized


def test_chat_message_gemini_error(test_client, auth_headers, mock_gemini_client):
    """Test chat message when Gemini client raises an exception."""
    # Mock Gemini client to raise exception
    mock_gemini_client.generate_text.side_effect = Exception("Gemini API error")
    
    # Make request
    response = test_client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": "Hello, how are you?"}
    )
    
    # Check response
    assert response.status_code == 500  # Internal Server Error
    assert "error" in response.json()
    assert "gemini api" in response.json()["error"].lower()


def test_chat_message_streaming(test_client, auth_headers, mock_gemini_client):
    """Test streaming chat message."""
    # Test data
    test_message = "Hello, how are you?"
    
    # Make request with streaming flag
    response = test_client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": test_message, "stream": True}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    
    # Parse SSE response
    response_text = response.text
    assert "data:" in response_text
    
    # Verify mock was called correctly
    mock_gemini_client.stream_text.assert_called_once_with(test_message, context=None)


def test_chat_message_streaming_with_history(test_client, auth_headers, mock_gemini_client):
    """Test streaming chat message with conversation history."""
    # Test data
    test_message = "What else can you tell me?"
    test_history = [
        {"role": "user", "content": "Tell me about fraud detection."},
        {"role": "assistant", "content": "Fraud detection involves identifying suspicious patterns."},
    ]
    
    # Make request with streaming flag and history
    response = test_client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": test_message, "history": test_history, "stream": True}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    
    # Verify mock was called correctly
    mock_gemini_client.stream_text.assert_called_once()
    # Check that history was included in the call
    call_args = mock_gemini_client.stream_text.call_args[1]
    assert "context" in call_args
    assert "fraud detection" in call_args["context"].lower()


def test_chat_message_streaming_error(test_client, auth_headers, mock_gemini_client):
    """Test streaming chat message when an error occurs."""
    # Mock Gemini client to raise exception
    mock_gemini_client.stream_text.side_effect = Exception("Streaming error")
    
    # Make request with streaming flag
    response = test_client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": "Hello, how are you?", "stream": True}
    )
    
    # Check response
    assert response.status_code == 500  # Internal Server Error
    assert "error" in response.json()
    assert "streaming error" in response.json()["error"].lower()


def test_chat_message_with_parameters(test_client, auth_headers, mock_gemini_client):
    """Test chat message with generation parameters."""
    # Test data
    test_message = "Hello, how are you?"
    test_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 500
    }
    
    # Make request with parameters
    response = test_client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": test_message, "parameters": test_params}
    )
    
    # Check response
    assert response.status_code == 200
    assert "response" in response.json()
    
    # Verify mock was called correctly with parameters
    # Note: The API should pass these parameters to the Gemini client
    mock_gemini_client.generate_text.assert_called_once()
