"""Unit tests for JWT authentication system.

This module contains tests for token creation, validation, refresh, and
authentication dependencies in the JWT authentication system.
"""

import time
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi import HTTPException, status, Request
from jose import jwt, JWTError

from backend.auth.jwt_handler import JWTHandler
from backend.auth.dependencies import (
    get_current_user,
    get_optional_user,
    require_roles,
    UserRole,
    rate_limit
)
from backend.config import settings


# ---- Fixtures ----

@pytest.fixture
def test_user_data():
    """Fixture for test user data."""
    return {
        "id": "test_user_id",
        "email": "test@example.com",
        "full_name": "Test User",
        "role": UserRole.ANALYST
    }


@pytest.fixture
def test_admin_data():
    """Fixture for test admin user data."""
    return {
        "id": "test_admin_id",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "role": UserRole.ADMIN
    }


@pytest.fixture
def access_token(test_user_data):
    """Fixture for a valid access token."""
    return JWTHandler.create_access_token(
        subject=test_user_data["id"],
        user_data=test_user_data
    )


@pytest.fixture
def refresh_token(test_user_data):
    """Fixture for a valid refresh token."""
    return JWTHandler.create_refresh_token(
        subject=test_user_data["id"]
    )


@pytest.fixture
def expired_token(test_user_data):
    """Fixture for an expired token."""
    # Create token with expiration in the past
    expire = datetime.utcnow() - timedelta(minutes=30)
    
    to_encode = {
        "sub": test_user_data["id"],
        "exp": expire,
        "iat": datetime.utcnow() - timedelta(minutes=60),
        "nbf": datetime.utcnow() - timedelta(minutes=60),
        "aud": settings.jwt_audience,
        "iss": settings.jwt_issuer,
        "jti": f"{test_user_data['id']}_{int(time.time())}"
    }
    
    return jwt.encode(
        to_encode, 
        settings.secret_key, 
        algorithm=settings.jwt_algorithm
    )


@pytest.fixture
def mock_request():
    """Fixture for a mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.headers = {}
    request.client.host = "127.0.0.1"
    request.state = MagicMock()
    return request


# ---- Tests for JWTHandler ----

def test_create_access_token(test_user_data):
    """Test creating an access token."""
    token = JWTHandler.create_access_token(
        subject=test_user_data["id"],
        user_data=test_user_data
    )
    
    # Verify token is a non-empty string
    assert isinstance(token, str)
    assert len(token) > 0
    
    # Decode and verify payload
    payload = jwt.decode(
        token,
        settings.secret_key,
        algorithms=[settings.jwt_algorithm],
        audience=settings.jwt_audience,
        issuer=settings.jwt_issuer
    )
    
    assert payload["sub"] == test_user_data["id"]
    assert "exp" in payload
    assert "iat" in payload
    assert "jti" in payload
    assert "user_data" in payload
    assert payload["user_data"]["email"] == test_user_data["email"]
    assert payload["user_data"]["role"] == test_user_data["role"]


def test_create_refresh_token(test_user_data):
    """Test creating a refresh token."""
    token = JWTHandler.create_refresh_token(
        subject=test_user_data["id"]
    )
    
    # Verify token is a non-empty string
    assert isinstance(token, str)
    assert len(token) > 0
    
    # Decode and verify payload
    payload = jwt.decode(
        token,
        settings.secret_key,
        algorithms=[settings.jwt_algorithm],
        audience=f"{settings.jwt_audience}:refresh",
        issuer=settings.jwt_issuer
    )
    
    assert payload["sub"] == test_user_data["id"]
    assert payload["type"] == "refresh"
    assert "exp" in payload
    assert "iat" in payload
    assert "jti" in payload


def test_decode_token(access_token, test_user_data):
    """Test decoding a valid access token."""
    payload = JWTHandler.decode_token(access_token)
    
    assert payload["sub"] == test_user_data["id"]
    assert "exp" in payload
    assert "user_data" in payload
    assert payload["user_data"]["email"] == test_user_data["email"]


def test_decode_token_invalid():
    """Test decoding an invalid token."""
    with pytest.raises(HTTPException) as exc_info:
        JWTHandler.decode_token("invalid.token.string")
    
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid token" in exc_info.value.detail


def test_decode_token_expired(expired_token):
    """Test decoding an expired token."""
    with pytest.raises(HTTPException) as exc_info:
        JWTHandler.decode_token(expired_token)
    
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid token" in exc_info.value.detail


def test_decode_refresh_token(refresh_token, test_user_data):
    """Test decoding a valid refresh token."""
    payload = JWTHandler.decode_refresh_token(refresh_token)
    
    assert payload["sub"] == test_user_data["id"]
    assert payload["type"] == "refresh"
    assert "exp" in payload


def test_decode_refresh_token_invalid():
    """Test decoding an invalid refresh token."""
    with pytest.raises(HTTPException) as exc_info:
        JWTHandler.decode_refresh_token("invalid.token.string")
    
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid refresh token" in exc_info.value.detail


def test_decode_refresh_token_wrong_type(access_token):
    """Test decoding an access token as refresh token."""
    with pytest.raises(HTTPException) as exc_info:
        JWTHandler.decode_refresh_token(access_token)
    
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


def test_refresh_tokens(refresh_token):
    """Test refreshing tokens."""
    with patch.object(JWTHandler, 'decode_refresh_token') as mock_decode:
        mock_decode.return_value = {"sub": "test_user_id", "type": "refresh"}
        
        with patch.object(JWTHandler, 'create_access_token') as mock_access:
            with patch.object(JWTHandler, 'create_refresh_token') as mock_refresh:
                mock_access.return_value = "new_access_token"
                mock_refresh.return_value = "new_refresh_token"
                
                access_token, new_refresh_token = JWTHandler.refresh_tokens(refresh_token)
                
                assert access_token == "new_access_token"
                assert new_refresh_token == "new_refresh_token"
                mock_decode.assert_called_once_with(refresh_token)
                mock_access.assert_called_once_with(subject="test_user_id")
                mock_refresh.assert_called_once_with(subject="test_user_id")


def test_verify_token_and_get_subject(access_token, test_user_data):
    """Test verifying token and extracting subject."""
    subject = JWTHandler.verify_token_and_get_subject(access_token)
    assert subject == test_user_data["id"]


def test_verify_token_and_get_subject_invalid():
    """Test verifying an invalid token."""
    with pytest.raises(HTTPException) as exc_info:
        JWTHandler.verify_token_and_get_subject("invalid.token.string")
    
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


# ---- Tests for Authentication Dependencies ----

@pytest.mark.asyncio
async def test_get_current_user_oauth2(access_token, test_user_data):
    """Test getting current user with OAuth2 token."""
    user = await get_current_user(token=access_token, credentials=None)
    
    assert user["id"] == test_user_data["id"]
    assert user["role"] == test_user_data["role"]
    assert "email" in user


@pytest.mark.asyncio
async def test_get_current_user_http_bearer(access_token, test_user_data):
    """Test getting current user with HTTP Bearer token."""
    # Create mock credentials
    credentials = MagicMock()
    credentials.credentials = access_token
    
    user = await get_current_user(token=None, credentials=credentials)
    
    assert user["id"] == test_user_data["id"]
    assert user["role"] == test_user_data["role"]


@pytest.mark.asyncio
async def test_get_current_user_no_token():
    """Test getting current user with no token."""
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=None, credentials=None)
    
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Not authenticated" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_user_invalid_token():
    """Test getting current user with invalid token."""
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token="invalid.token.string", credentials=None)
    
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_get_optional_user_with_token(access_token, test_user_data):
    """Test getting optional user with valid token."""
    # Mock the get_current_user dependency
    async def mock_get_current_user(*args, **kwargs):
        return test_user_data
    
    user = await get_optional_user(current_user=await mock_get_current_user())
    
    assert user == test_user_data


@pytest.mark.asyncio
async def test_get_optional_user_no_token():
    """Test getting optional user with no token."""
    user = await get_optional_user(current_user=None)
    assert user is None


@pytest.mark.asyncio
async def test_require_roles_allowed(test_admin_data):
    """Test role requirement when user has allowed role."""
    # Create role checker for admin role
    role_checker = require_roles([UserRole.ADMIN])
    
    # Call the role checker with admin user
    user = await role_checker(current_user=test_admin_data)
    
    assert user == test_admin_data


@pytest.mark.asyncio
async def test_require_roles_not_allowed(test_user_data):
    """Test role requirement when user doesn't have allowed role."""
    # Create role checker for admin role
    role_checker = require_roles([UserRole.ADMIN])
    
    # Call the role checker with non-admin user
    with pytest.raises(HTTPException) as exc_info:
        await role_checker(current_user=test_user_data)
    
    assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
    assert "Insufficient permissions" in exc_info.value.detail


@pytest.mark.asyncio
async def test_rate_limit_under_limit(mock_request):
    """Test rate limiting when under the limit."""
    # Create rate limiter
    limiter = rate_limit(requests_per_minute=10)
    
    # Call the rate limiter multiple times (under limit)
    for _ in range(5):
        result = await limiter(request=mock_request)
        assert result is True


@pytest.mark.asyncio
async def test_rate_limit_exceeded(mock_request):
    """Test rate limiting when limit exceeded."""
    # Create rate limiter with low limit
    limiter = rate_limit(requests_per_minute=3)
    
    # Call the rate limiter multiple times (over limit)
    for _ in range(3):
        result = await limiter(request=mock_request)
        assert result is True
    
    # Next call should raise exception
    with pytest.raises(HTTPException) as exc_info:
        await limiter(request=mock_request)
    
    assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "Rate limit exceeded" in exc_info.value.detail
