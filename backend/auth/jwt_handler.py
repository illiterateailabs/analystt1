"""JWT token handling for the Analyst's Augmentation Agent.

This module provides functionality for creating, validating, and refreshing JWT tokens
for authentication and authorization purposes.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple, Set

from fastapi import HTTPException, status
from jose import jwt, JWTError
from pydantic import ValidationError

from backend.config import settings

# Import Redis for blacklist persistence
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis.asyncio not available, falling back to in-memory blacklist")


logger = logging.getLogger(__name__)


class JWTHandler:
    """Handler for JWT token operations."""
    
    # Class-level variables
    _redis_client = None
    _blacklist: Set[str] = set()  # In-memory fallback
    _initialized = False

    @classmethod
    async def init(cls, redis_url: Optional[str] = None):
        """
        Initialize the JWT handler with Redis connection.
        
        Args:
            redis_url: Redis connection URL, defaults to settings
        """
        if cls._initialized:
            return
            
        # Set up Redis connection if available
        if REDIS_AVAILABLE:
            try:
                redis_url = redis_url or settings.REDIS_URL
                cls._redis_client = redis.Redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_timeout=2.0  # Short timeout to fail fast
                )
                # Test connection
                await cls._redis_client.ping()
                logger.info("Connected to Redis for JWT blacklist")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory blacklist.")
                cls._redis_client = None
        
        cls._initialized = True

    @classmethod
    async def is_token_blacklisted(cls, token_id: str) -> bool:
        """
        Check if a token is blacklisted.
        
        Args:
            token_id: JWT token ID (jti claim)
            
        Returns:
            True if token is blacklisted, False otherwise
        """
        # Ensure initialized
        if not cls._initialized:
            await cls.init()
            
        # Try Redis first if available
        if cls._redis_client:
            try:
                exists = await cls._redis_client.exists(f"blacklist:{token_id}")
                return bool(exists)
            except Exception as e:
                logger.warning(f"Redis error checking blacklist: {e}. Falling back to in-memory.")
                
        # Fallback to in-memory blacklist
        return token_id in cls._blacklist

    @classmethod
    async def blacklist_token(cls, token_id: str, expires_at: Optional[datetime] = None):
        """
        Add a token to the blacklist.
        
        Args:
            token_id: JWT token ID (jti claim)
            expires_at: When the token expires (for TTL)
        """
        # Ensure initialized
        if not cls._initialized:
            await cls.init()
            
        # Calculate TTL in seconds
        ttl = None
        if expires_at:
            now = datetime.utcnow()
            if expires_at > now:
                ttl = int((expires_at - now).total_seconds())
        
        # Try Redis first if available
        if cls._redis_client:
            try:
                await cls._redis_client.set(f"blacklist:{token_id}", "1")
                
                # Set expiry if provided
                if ttl:
                    await cls._redis_client.expire(f"blacklist:{token_id}", ttl)
                    
                logger.debug(f"Token {token_id} blacklisted in Redis with TTL: {ttl}")
                return
            except Exception as e:
                logger.warning(f"Redis error adding to blacklist: {e}. Falling back to in-memory.")
        
        # Fallback to in-memory blacklist
        cls._blacklist.add(token_id)
        logger.debug(f"Token {token_id} blacklisted in memory")
        
        # We can't set TTL for in-memory blacklist, so we'll need periodic cleanup
        # This is a limitation of the fallback approach

    @classmethod
    async def revoke_token(cls, token: str):
        """
        Explicitly revoke a token by adding it to the blacklist.
        
        Args:
            token: JWT token to revoke
            
        Returns:
            True if token was revoked, False otherwise
        """
        try:
            # Decode the token without verification to get the jti and exp
            # This allows revoking even if the token is already expired
            payload = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False}
            )
            
            token_id = payload.get("jti")
            if not token_id:
                logger.warning("Cannot revoke token: missing jti claim")
                return False
                
            # Get expiration time if available
            expires_at = None
            if "exp" in payload:
                expires_at = datetime.fromtimestamp(payload["exp"])
                
            # Add to blacklist
            await cls.blacklist_token(token_id, expires_at)
            return True
            
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False

    @classmethod
    async def cleanup(cls):
        """Close Redis connection and clear in-memory blacklist."""
        if cls._redis_client:
            try:
                await cls._redis_client.close()
                logger.info("Closed Redis connection for JWT blacklist")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
                
        cls._blacklist.clear()
        cls._initialized = False

    @staticmethod
    def create_access_token(
        subject: str,
        user_data: Dict[str, Any] = None,
        expires_delta: Optional[int] = None,
    ) -> str:
        """
        Create a new JWT access token.

        Args:
            subject: Token subject (usually user ID)
            user_data: Additional user data to include in token
            expires_delta: Optional custom expiration time in minutes

        Returns:
            JWT token string
        """
        if expires_delta is not None:
            expires_minutes = expires_delta
        else:
            expires_minutes = settings.JWT_EXPIRATION_MINUTES

        expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
        
        to_encode = {
            "sub": str(subject),
            "exp": expire,
            "iat": datetime.utcnow(),
            "nbf": datetime.utcnow(),
            "aud": settings.JWT_AUDIENCE,
            "iss": settings.JWT_ISSUER,
            "jti": f"{subject}_{int(time.time())}"
        }
        
        # Add user data if provided
        if user_data:
            to_encode.update({"user_data": user_data})
            
        # Create token
        encoded_jwt = jwt.encode(
            to_encode,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        
        return encoded_jwt

    @staticmethod
    def create_refresh_token(
        subject: str,
        expires_delta: Optional[int] = None,
    ) -> str:
        """
        Create a new JWT refresh token.

        Args:
            subject: Token subject (usually user ID)
            expires_delta: Optional custom expiration time in minutes

        Returns:
            JWT refresh token string
        """
        # Refresh tokens typically have longer expiration
        if expires_delta is not None:
            expires_minutes = expires_delta
        else:
            # Default to 7 days for refresh tokens
            expires_minutes = 60 * 24 * 7
            
        expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
        
        to_encode = {
            "sub": str(subject),
            "exp": expire,
            "iat": datetime.utcnow(),
            "nbf": datetime.utcnow(),
            "aud": f"{settings.JWT_AUDIENCE}:refresh",
            "iss": settings.JWT_ISSUER,
            "jti": f"refresh_{subject}_{int(time.time())}",
            "type": "refresh"
        }
            
        # Create token
        encoded_jwt = jwt.encode(
            to_encode,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        
        return encoded_jwt

    @classmethod
    async def decode_token(cls, token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT token to decode

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid, expired, or blacklisted
        """
        try:
            # First decode without verification to check blacklist
            unverified_payload = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            
            # Check if token is blacklisted
            token_id = unverified_payload.get("jti")
            if token_id and await cls.is_token_blacklisted(token_id):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Now verify the token
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
                audience=settings.JWT_AUDIENCE,
                issuer=settings.JWT_ISSUER,
            )
            return payload
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except ValidationError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @classmethod
    async def decode_refresh_token(cls, token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT refresh token.

        Args:
            token: JWT refresh token to decode

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid, expired, blacklisted, or not a refresh token
        """
        try:
            # First decode without verification to check blacklist
            unverified_payload = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            
            # Check if token is blacklisted
            token_id = unverified_payload.get("jti")
            if token_id and await cls.is_token_blacklisted(token_id):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Refresh token has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Now verify the token
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
                audience=f"{settings.JWT_AUDIENCE}:refresh",
                issuer=settings.JWT_ISSUER,
            )
            
            # Verify this is a refresh token
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type: not a refresh token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
            return payload
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid refresh token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except ValidationError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @classmethod
    async def refresh_tokens(cls, refresh_token: str) -> Tuple[str, str]:
        """
        Generate new access and refresh tokens using a valid refresh token.
        Also blacklists the old refresh token for security.

        Args:
            refresh_token: Valid refresh token

        Returns:
            Tuple of (new_access_token, new_refresh_token)

        Raises:
            HTTPException: If refresh token is invalid or expired
        """
        # Decode and validate the refresh token
        payload = await cls.decode_refresh_token(refresh_token)
        
        # Extract subject from payload
        subject = payload.get("sub")
        if not subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Blacklist the old refresh token (security best practice)
        token_id = payload.get("jti")
        if token_id:
            await cls.blacklist_token(token_id)
            
        # Create new tokens
        new_access_token = cls.create_access_token(subject=subject)
        new_refresh_token = cls.create_refresh_token(subject=subject)
        
        return new_access_token, new_refresh_token

    @classmethod
    async def verify_token_and_get_subject(cls, token: str) -> str:
        """
        Verify token and extract subject.

        Args:
            token: JWT token to verify

        Returns:
            Subject from token (usually user ID)

        Raises:
            HTTPException: If token is invalid, expired, or blacklisted
        """
        payload = await cls.decode_token(token)
        
        subject = payload.get("sub")
        if not subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return subject


# --------------------------------------------------------------------------- #
# Convenience wrapper functions
# --------------------------------------------------------------------------- #
# The new cookie-based auth layer (secure_cookies.py) imports the helpers
# below directly. They provide a thin façade over the JWTHandler class so that
# external modules do not have to deal with the class API change.

import asyncio


def _run_sync(coro):
    """
    Run *coro* in the current event-loop if one is running, otherwise start a
    temporary loop.  This allows synchronous helper functions to call async
    JWTHandler methods safely from both sync and async contexts.
    """
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
    except RuntimeError:
        # No event-loop running in this thread
        pass
    return asyncio.run(coro)


def create_access_token(data: Dict[str, Any], expires_delta: datetime) -> str:  # noqa: N802
    """
    Backwards-compat helper expected by *secure_cookies.py*.

    Args:
        data: Full JWT payload; **must include** ``sub`` claim.
        expires_delta: Absolute UTC expiry (``datetime``).

    Returns:
        Encoded JWT string.
    """
    subject = data.get("sub")
    if subject is None:
        raise ValueError("`sub` (subject) claim missing from token data")

    # Remaining claims become ``user_data`` for JWTHandler
    user_data = data.copy()
    user_data.pop("sub", None)

    # Convert absolute expiry → minutes from now expected by JWTHandler
    delta_minutes = max(
        1,
        int((expires_delta - datetime.utcnow()).total_seconds() // 60),
    )
    return JWTHandler.create_access_token(
        subject=subject,
        user_data=user_data,
        expires_delta=delta_minutes,
    )


def decode_jwt(token: str) -> Dict[str, Any]:  # noqa: N802
    """
    Synchronous helper that validates and decodes a JWT.
    """
    return _run_sync(JWTHandler.decode_token(token))


async def add_token_to_blacklist(token: str, token_id: str):  # noqa: N802
    """
    Async helper used by secure_cookies.py when rotating refresh tokens.

    Args:
        token: Full JWT string (only used to derive expiry for TTL).
        token_id: The **jti** claim identifying the token to revoke.
    """
    # Determine expiry (optional TTL for Redis)
    expires_at = None
    try:
        unverified = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
        if "exp" in unverified:
            expires_at = datetime.fromtimestamp(unverified["exp"])
    except Exception:
        # If we can't parse the token we still blacklist without TTL
        pass

    await JWTHandler.blacklist_token(token_id, expires_at)

