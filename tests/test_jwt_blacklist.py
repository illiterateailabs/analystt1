"""
Tests for JWT blacklist functionality in JWTHandler.

This module tests the JWT token blacklist implementation, including:
- Initialization with and without Redis
- Token blacklisting and checking
- Token revocation
- TTL functionality
- Fallback to in-memory when Redis is unavailable
- Edge cases and error handling
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from jose import JWTError # Added missing import
from fastapi import HTTPException # Added missing import

from backend.auth.jwt_handler import JWTHandler, REDIS_AVAILABLE
from backend.config import settings


# Helper to create a dummy token payload for testing revocation
def create_dummy_token_payload(jti: str, exp_delta_seconds: int = 3600) -> dict:
    now = datetime.utcnow()
    return {
        "jti": jti,
        "exp": int((now + timedelta(seconds=exp_delta_seconds)).timestamp()),
        "sub": "test_user",
        "iat": int(now.timestamp())
    }

# Fixture to ensure JWTHandler is cleaned up and re-initialized for each test
@pytest.fixture(autouse=True)
async def cleanup_jwt_handler():
    # Reset initialization state before each test
    JWTHandler._initialized = False
    JWTHandler._redis_client = None
    JWTHandler._blacklist.clear()
    yield
    # Cleanup after each test
    await JWTHandler.cleanup()
    JWTHandler._initialized = False # Ensure it's reset for the next test
    JWTHandler._redis_client = None
    JWTHandler._blacklist.clear()


@pytest.mark.asyncio
async def test_jwt_handler_init_with_redis_success():
    """Test JWTHandler initialization with a working Redis connection."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis not available, skipping Redis-specific test")

    mock_redis_instance = AsyncMock()
    mock_redis_instance.ping = AsyncMock(return_value=True)

    with patch("redis.asyncio.Redis.from_url", return_value=mock_redis_instance) as mock_from_url:
        await JWTHandler.init(redis_url="redis://localhost:6379/0")
        mock_from_url.assert_called_once_with(
            "redis://localhost:6379/0",
            decode_responses=True,
            socket_timeout=2.0
        )
        mock_redis_instance.ping.assert_called_once()
        assert JWTHandler._redis_client == mock_redis_instance
        assert JWTHandler._initialized is True

@pytest.mark.asyncio
async def test_jwt_handler_init_with_redis_failure_falls_back_to_in_memory():
    """Test JWTHandler falls back to in-memory if Redis connection fails."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis not available, skipping Redis-specific test")

    with patch("redis.asyncio.Redis.from_url", side_effect=Exception("Redis connection error")) as mock_from_url:
        await JWTHandler.init(redis_url="redis://localhost:6379/0")
        mock_from_url.assert_called_once()
        assert JWTHandler._redis_client is None
        assert JWTHandler._initialized is True # Should still be initialized for in-memory
        # Check if a warning was logged (optional, depends on logging setup for tests)

@pytest.mark.asyncio
async def test_jwt_handler_init_without_redis_module():
    """Test JWTHandler initialization when redis module is not available."""
    with patch("backend.auth.jwt_handler.REDIS_AVAILABLE", False):
        # Re-import or re-evaluate JWTHandler if REDIS_AVAILABLE is checked at import time
        # For this test, we assume it's checked dynamically or we can force re-init
        JWTHandler._initialized = False # Force re-init
        await JWTHandler.init()
        assert JWTHandler._redis_client is None
        assert JWTHandler._initialized is True
        assert len(JWTHandler._blacklist) == 0


@pytest.mark.asyncio
async def test_blacklist_token_and_check_with_redis():
    """Test blacklisting and checking a token with Redis."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis not available, skipping Redis-specific test")

    mock_redis_instance = AsyncMock()
    mock_redis_instance.ping = AsyncMock(return_value=True)
    mock_redis_instance.set = AsyncMock()
    mock_redis_instance.expire = AsyncMock()
    mock_redis_instance.exists = AsyncMock(return_value=True)

    with patch("redis.asyncio.Redis.from_url", return_value=mock_redis_instance):
        await JWTHandler.init() # Initialize with mocked Redis
        
        token_id = "test_jti_1"
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        await JWTHandler.blacklist_token(token_id, expires_at)
        
        mock_redis_instance.set.assert_called_once_with(f"blacklist:{token_id}", "1")
        ttl_seconds = int((expires_at - datetime.utcnow()).total_seconds())
        # Allow for slight timing difference in ttl calculation
        mock_redis_instance.expire.assert_called_once()
        args, _ = mock_redis_instance.expire.call_args
        assert args[0] == f"blacklist:{token_id}"
        assert abs(args[1] - ttl_seconds) <= 1 # ttl should be close
        
        is_blacklisted = await JWTHandler.is_token_blacklisted(token_id)
        mock_redis_instance.exists.assert_called_once_with(f"blacklist:{token_id}")
        assert is_blacklisted is True

@pytest.mark.asyncio
async def test_blacklist_token_and_check_in_memory():
    """Test blacklisting and checking a token with in-memory fallback."""
    # Force in-memory by making Redis connection fail
    with patch("redis.asyncio.Redis.from_url", side_effect=Exception("Redis down")):
        await JWTHandler.init()
        assert JWTHandler._redis_client is None

        token_id = "test_jti_memory_1"
        await JWTHandler.blacklist_token(token_id)
        assert token_id in JWTHandler._blacklist
        
        is_blacklisted = await JWTHandler.is_token_blacklisted(token_id)
        assert is_blacklisted is True

        is_not_blacklisted = await JWTHandler.is_token_blacklisted("non_existent_jti")
        assert is_not_blacklisted is False

@pytest.mark.asyncio
async def test_revoke_token_success():
    """Test successful token revocation."""
    token_id = "revoke_jti_1"
    # Create a dummy token string (payload doesn't matter as much as structure for decode)
    # For simplicity, we'll mock the decode part of revoke_token
    
    mock_payload = create_dummy_token_payload(token_id)

    with patch("jose.jwt.decode", return_value=mock_payload), \
         patch.object(JWTHandler, "blacklist_token", new_callable=AsyncMock) as mock_blacklist:
        
        # The actual token string content doesn't matter here due to mocking jwt.decode
        dummy_token_str = "dummy.jwt.token"
        revoked = await JWTHandler.revoke_token(dummy_token_str)
        
        assert revoked is True
        mock_blacklist.assert_called_once()
        # Check that blacklist_token was called with the correct jti and expiry
        args, _ = mock_blacklist.call_args
        assert args[0] == token_id
        assert isinstance(args[1], datetime) # expires_at should be a datetime object


@pytest.mark.asyncio
async def test_revoke_token_missing_jti():
    """Test token revocation failure if JTI is missing."""
    mock_payload = {"sub": "test_user"} # No JTI
    with patch("jose.jwt.decode", return_value=mock_payload), \
         patch.object(JWTHandler, "blacklist_token", new_callable=AsyncMock) as mock_blacklist:
        
        dummy_token_str = "dummy.jwt.token.no.jti"
        revoked = await JWTHandler.revoke_token(dummy_token_str)
        
        assert revoked is False
        mock_blacklist.assert_not_called()

@pytest.mark.asyncio
async def test_revoke_token_jwt_decode_error():
    """Test token revocation failure if token decoding fails."""
    with patch("jose.jwt.decode", side_effect=JWTError("Invalid token")), \
         patch.object(JWTHandler, "blacklist_token", new_callable=AsyncMock) as mock_blacklist:
        
        dummy_token_str = "invalid.jwt.token"
        revoked = await JWTHandler.revoke_token(dummy_token_str)
        
        assert revoked is False
        mock_blacklist.assert_not_called()


@pytest.mark.asyncio
async def test_ttl_functionality_with_redis():
    """Test TTL is set correctly when blacklisting with Redis."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis not available, skipping Redis-specific test")

    mock_redis_instance = AsyncMock()
    mock_redis_instance.ping = AsyncMock(return_value=True)
    mock_redis_instance.set = AsyncMock()
    mock_redis_instance.expire = AsyncMock()

    with patch("redis.asyncio.Redis.from_url", return_value=mock_redis_instance):
        await JWTHandler.init()
        
        token_id = "test_jti_ttl"
        # Token expires in 10 seconds
        expires_at = datetime.utcnow() + timedelta(seconds=10)
        
        await JWTHandler.blacklist_token(token_id, expires_at)
        
        mock_redis_instance.expire.assert_called_once()
        args, _ = mock_redis_instance.expire.call_args
        assert args[0] == f"blacklist:{token_id}"
        # TTL should be around 10 seconds
        assert 8 <= args[1] <= 10 # Allow for slight delay in execution

        # Test with already expired token (no TTL or negative TTL)
        mock_redis_instance.expire.reset_mock()
        expired_token_id = "test_jti_expired_ttl"
        past_expires_at = datetime.utcnow() - timedelta(seconds=10)
        await JWTHandler.blacklist_token(expired_token_id, past_expires_at)
        # Expire should not be called if TTL is not positive
        # Or, if Redis client handles negative TTL by not setting, that's fine too.
        # Current implementation calculates positive TTL, so if expires_at < now, ttl is None
        mock_redis_instance.expire.assert_not_called() 
        # Or if it's called with 0 or negative, that's also acceptable depending on redis client behavior
        # For current code: if ttl is None, expire is not called.

@pytest.mark.asyncio
async def test_is_token_blacklisted_redis_error_falls_back_to_in_memory():
    """Test is_token_blacklisted falls back to in-memory if Redis errors."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis not available, skipping Redis-specific test")

    mock_redis_instance = AsyncMock()
    mock_redis_instance.ping = AsyncMock(return_value=True)
    # Simulate Redis error on .exists()
    mock_redis_instance.exists = AsyncMock(side_effect=Exception("Redis connection error"))

    with patch("redis.asyncio.Redis.from_url", return_value=mock_redis_instance):
        await JWTHandler.init()
        
        token_id_in_memory = "jti_mem_fallback_check"
        JWTHandler._blacklist.add(token_id_in_memory) # Manually add to in-memory
        
        is_blacklisted = await JWTHandler.is_token_blacklisted(token_id_in_memory)
        assert is_blacklisted is True # Should find it in memory
        mock_redis_instance.exists.assert_called_once_with(f"blacklist:{token_id_in_memory}")

@pytest.mark.asyncio
async def test_blacklist_token_redis_error_falls_back_to_in_memory():
    """Test blacklist_token falls back to in-memory if Redis errors."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis not available, skipping Redis-specific test")

    mock_redis_instance = AsyncMock()
    mock_redis_instance.ping = AsyncMock(return_value=True)
    # Simulate Redis error on .set()
    mock_redis_instance.set = AsyncMock(side_effect=Exception("Redis connection error"))

    with patch("redis.asyncio.Redis.from_url", return_value=mock_redis_instance):
        await JWTHandler.init()
        
        token_id_to_blacklist = "jti_mem_fallback_add"
        expires_at = datetime.utcnow() + timedelta(hours=1)
        
        await JWTHandler.blacklist_token(token_id_to_blacklist, expires_at)
        
        mock_redis_instance.set.assert_called_once_with(f"blacklist:{token_id_to_blacklist}", "1")
        assert token_id_to_blacklist in JWTHandler._blacklist # Should be added to in-memory

@pytest.mark.asyncio
async def test_decode_token_checks_blacklist():
    """Test that decode_token checks the blacklist."""
    token_id = "decode_check_jti"
    mock_unverified_payload = create_dummy_token_payload(token_id)
    # This payload will be returned by the final jwt.decode if not blacklisted
    mock_verified_payload = {**mock_unverified_payload, "aud": settings.jwt_audience, "iss": settings.jwt_issuer}

    with patch("jose.jwt.decode") as mock_jwt_decode, \
         patch.object(JWTHandler, "is_token_blacklisted", new_callable=AsyncMock) as mock_is_blacklisted:

        # Scenario 1: Token is blacklisted
        mock_is_blacklisted.return_value = True
        # First call to jwt.decode is for unverified payload, second for verified
        mock_jwt_decode.side_effect = [mock_unverified_payload, JWTError("Should not be called")] 
        
        with pytest.raises(HTTPException) as exc_info:
            await JWTHandler.decode_token("blacklisted.dummy.token")
        assert exc_info.value.status_code == 401
        assert "Token has been revoked" in exc_info.value.detail
        mock_is_blacklisted.assert_called_once_with(token_id)
        assert mock_jwt_decode.call_count == 1 # Only unverified decode should happen
        
        # Scenario 2: Token is NOT blacklisted
        mock_is_blacklisted.reset_mock()
        mock_jwt_decode.reset_mock()
        mock_is_blacklisted.return_value = False
        mock_jwt_decode.side_effect = [mock_unverified_payload, mock_verified_payload]

        decoded = await JWTHandler.decode_token("valid.dummy.token")
        assert decoded == mock_verified_payload
        mock_is_blacklisted.assert_called_once_with(token_id)
        assert mock_jwt_decode.call_count == 2 # Both unverified and verified decodes

@pytest.mark.asyncio
async def test_decode_refresh_token_checks_blacklist():
    """Test that decode_refresh_token checks the blacklist."""
    token_id = "decode_refresh_check_jti"
    mock_unverified_payload = {**create_dummy_token_payload(token_id), "type": "refresh"}
    mock_verified_payload = {**mock_unverified_payload, "aud": f"{settings.jwt_audience}:refresh", "iss": settings.jwt_issuer}

    with patch("jose.jwt.decode") as mock_jwt_decode, \
         patch.object(JWTHandler, "is_token_blacklisted", new_callable=AsyncMock) as mock_is_blacklisted:

        # Scenario 1: Token is blacklisted
        mock_is_blacklisted.return_value = True
        mock_jwt_decode.side_effect = [mock_unverified_payload, JWTError("Should not be called")]
        
        with pytest.raises(HTTPException) as exc_info:
            await JWTHandler.decode_refresh_token("blacklisted.refresh.token")
        assert exc_info.value.status_code == 401
        assert "Refresh token has been revoked" in exc_info.value.detail
        
        # Scenario 2: Token is NOT blacklisted
        mock_is_blacklisted.reset_mock()
        mock_jwt_decode.reset_mock()
        mock_is_blacklisted.return_value = False
        mock_jwt_decode.side_effect = [mock_unverified_payload, mock_verified_payload]

        decoded = await JWTHandler.decode_refresh_token("valid.refresh.token")
        assert decoded == mock_verified_payload


@pytest.mark.asyncio
async def test_refresh_tokens_blacklists_old_refresh_token():
    """Test that refresh_tokens blacklists the old refresh token."""
    old_refresh_token_id = "old_refresh_jti"
    old_refresh_payload = {
        **create_dummy_token_payload(old_refresh_token_id), 
        "type": "refresh",
        "aud": f"{settings.jwt_audience}:refresh", # Needed for decode_refresh_token
        "iss": settings.jwt_issuer, # Needed for decode_refresh_token
    }

    # Mock decode_refresh_token to return our specific payload
    # Mock create_access_token and create_refresh_token
    # Mock blacklist_token to verify it's called
    with patch.object(JWTHandler, "decode_refresh_token", new_callable=AsyncMock, return_value=old_refresh_payload) as mock_decode, \
         patch.object(JWTHandler, "create_access_token", return_value="new.access.token") as mock_create_access, \
         patch.object(JWTHandler, "create_refresh_token", return_value="new.refresh.token") as mock_create_refresh, \
         patch.object(JWTHandler, "blacklist_token", new_callable=AsyncMock) as mock_blacklist:
        
        new_access, new_refresh = await JWTHandler.refresh_tokens("old.refresh.token.string")

        assert new_access == "new.access.token"
        assert new_refresh == "new.refresh.token"
        
        mock_decode.assert_called_once_with("old.refresh.token.string")
        mock_create_access.assert_called_once_with(subject=old_refresh_payload["sub"])
        mock_create_refresh.assert_called_once_with(subject=old_refresh_payload["sub"])
        mock_blacklist.assert_called_once_with(old_refresh_token_id)

@pytest.mark.asyncio
async def test_cleanup_closes_redis_and_clears_blacklist():
    """Test that cleanup method closes Redis and clears in-memory blacklist."""
    mock_redis_instance = AsyncMock()
    mock_redis_instance.ping = AsyncMock(return_value=True)
    mock_redis_instance.close = AsyncMock()

    with patch("redis.asyncio.Redis.from_url", return_value=mock_redis_instance):
        await JWTHandler.init() # Initialize with mocked Redis
        JWTHandler._blacklist.add("some_jti_in_memory")
        assert JWTHandler._redis_client is not None
        assert len(JWTHandler._blacklist) > 0
        assert JWTHandler._initialized is True

        await JWTHandler.cleanup()

        if REDIS_AVAILABLE:
            mock_redis_instance.close.assert_called_once()
        assert len(JWTHandler._blacklist) == 0
        assert JWTHandler._initialized is False # Cleanup should reset this
        assert JWTHandler._redis_client is None # Should be reset by cleanup

# Test case for when redis_url is None in settings, should use default from settings
@pytest.mark.asyncio
async def test_jwt_handler_init_with_default_redis_url_from_settings():
    if not REDIS_AVAILABLE:
        pytest.skip("Redis not available")

    mock_redis_instance = AsyncMock()
    mock_redis_instance.ping = AsyncMock(return_value=True)
    
    # Temporarily modify settings for this test
    original_redis_url = settings.redis_url
    settings.redis_url = "redis://default-from-settings:6379/1"

    with patch("redis.asyncio.Redis.from_url", return_value=mock_redis_instance) as mock_from_url:
        await JWTHandler.init() # Call init without explicit redis_url
        mock_from_url.assert_called_once_with(
            "redis://default-from-settings:6379/1", # Ensure it used the settings value
            decode_responses=True,
            socket_timeout=2.0
        )
        assert JWTHandler._redis_client is not None
    
    settings.redis_url = original_redis_url # Restore original settings

