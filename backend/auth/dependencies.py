"""FastAPI dependencies for authentication and authorization.

This module provides dependencies for JWT token validation, optional authentication,
role-based access control, and rate limiting for FastAPI routes.
"""

import time
from enum import Enum
from typing import Dict, List, Optional, Union, Callable

from fastapi import Depends, HTTPException, Request, status, Cookie
from fastapi.security import OAuth2PasswordBearer, HTTPBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from backend.auth.jwt_handler import decode_jwt
from backend.auth.secure_cookies import (
    get_token_from_cookies_or_header,
    verify_csrf_token,
    ACCESS_TOKEN_COOKIE,
    CSRF_TOKEN_COOKIE
)
from backend.database import get_db
from backend.models.user import User
from backend.config import settings

# OAuth2 scheme for token extraction from requests (backward compatibility)
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    auto_error=False  # Don't auto-raise for optional auth
)

# HTTP Bearer scheme (alternative to OAuth2, backward compatibility)
http_bearer = HTTPBearer(auto_error=False)


class UserRole(str, Enum):
    """
    Simplified role enumeration.

    The `User` SQLAlchemy model only contains an `is_superuser` flag – there is
    no per-user role column.  We therefore reduce RBAC checks to two effective
    permission levels:

    * ``ADMIN`` → `user.is_superuser` is **True**
    * ``USER``  → any authenticated (non-admin) account

    Extra roles that previously existed (e.g. *analyst*, *viewer*) have been
    collapsed into the generic *USER* level to avoid referencing a non-existent
    `role` attribute.
    """

    ADMIN = "admin"
    USER = "user"


# Simple in-memory rate limiting store
# In production, use Redis or similar for distributed rate limiting
_rate_limit_store: Dict[str, Dict[str, Union[int, float]]] = {}


async def get_current_user(
    request: Request,
    access_token: Optional[str] = Cookie(None, alias=ACCESS_TOKEN_COOKIE),
    csrf_token: Optional[str] = Cookie(None, alias=CSRF_TOKEN_COOKIE),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Validate authentication and return current user.
    
    This dependency extracts and validates the authentication token from cookies
    or Authorization header (for backward compatibility), then loads the user
    from the database.
    
    Args:
        request: FastAPI request object
        access_token: Access token from cookies
        csrf_token: CSRF token from cookies
        db: Database session
        
    Returns:
        User object for the authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Get token using the secure cookies system (with fallback to Bearer)
        token = await get_token_from_cookies_or_header(
            request=request,
            access_token=access_token,
            csrf_token=csrf_token
        )
        
        # For non-GET requests, verify CSRF token
        if request.method not in ["GET", "HEAD", "OPTIONS"]:
            verify_csrf_token(request, csrf_token)
        
        # Decode and validate token
        payload = decode_jwt(token)
        
        # Extract user ID
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user identifier",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Store user in request state for easy access
        request.state.user = user
        
        return user
        
    except (JWTError, HTTPException) as e:
        if isinstance(e, HTTPException):
            raise e
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    request: Request,
    current_user: Optional[User] = Depends(get_current_user)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.
    
    This dependency is useful for endpoints that can work with or without authentication.
    
    Args:
        request: FastAPI request object
        current_user: User from token validation dependency
        
    Returns:
        User object if authenticated, None otherwise
    """
    try:
        return current_user
    except HTTPException:
        return None


def require_roles(allowed_roles: List[UserRole]):
    """
    Create a dependency that requires specific roles.
    
    Args:
        allowed_roles: List of roles that are allowed to access the endpoint
        
    Returns:
        Dependency function that validates user roles
    """
    async def role_checker(current_user: User = Depends(get_current_user)):
        """
        Validate user's role against the required roles.
        
        If ADMIN role is the only role required, only superusers are allowed.
        If USER role is included in allowed_roles, any authenticated user is allowed.
        """
        # If only ADMIN role is required, ensure the user is a superuser
        if UserRole.ADMIN in allowed_roles and UserRole.USER not in allowed_roles:
            if not current_user.is_superuser:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin privileges required",
                )
        
        # If we got here, the user is authenticated and has the required role
        # (either they're an admin when admin is required, or USER role is allowed)
        return current_user
    
    return role_checker


# Convenience role dependencies
# Convenience dependencies after simplification
require_admin = require_roles([UserRole.ADMIN])
# Generic authenticated user (non-admin) – use in place of analyst/viewer
require_authenticated = require_roles([UserRole.USER])


def rate_limit(
    requests_per_minute: int = 60,
    key_func: Callable[[Request], str] = None
):
    """
    Create a rate limiting dependency.
    
    Args:
        requests_per_minute: Maximum number of requests allowed per minute
        key_func: Function to extract rate limit key from request (defaults to IP address)
        
    Returns:
        Dependency function that enforces rate limits
    """
    async def rate_limiter(request: Request):
        # Get rate limit key (default to client IP)
        if key_func:
            key = key_func(request)
        else:
            # Get client IP, considering forwarded headers
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                key = forwarded.split(",")[0].strip()
            else:
                key = request.client.host
        
        # Get current time
        current_time = time.time()
        
        # Initialize or get rate limit data for this key
        if key not in _rate_limit_store:
            _rate_limit_store[key] = {
                "requests": 0,
                "window_start": current_time
            }
        
        # Reset counter if window has expired (1 minute)
        if current_time - _rate_limit_store[key]["window_start"] > 60:
            _rate_limit_store[key] = {
                "requests": 1,
                "window_start": current_time
            }
        else:
            # Increment request count
            _rate_limit_store[key]["requests"] += 1
        
        # Check if rate limit exceeded
        if _rate_limit_store[key]["requests"] > requests_per_minute:
            # Calculate retry-after time
            retry_after = int(60 - (current_time - _rate_limit_store[key]["window_start"]))
            
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(max(1, retry_after)),
                    "X-RateLimit-Limit": str(requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(_rate_limit_store[key]["window_start"] + 60))
                }
            )
        
        # Add rate limit headers to response
        request.state.rate_limit = {
            "limit": requests_per_minute,
            "remaining": requests_per_minute - _rate_limit_store[key]["requests"],
            "reset": int(_rate_limit_store[key]["window_start"] + 60)
        }
        
        return True
    
    return rate_limiter


# Rate limit by authenticated user ID
def get_user_rate_limit_key(request: Request) -> str:
    """
    Get rate limit key based on authenticated user ID.
    
    Falls back to IP address if user is not authenticated.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Rate limit key (user ID or IP address)
    """
    # Try to get user from request state (set by auth middleware)
    user = getattr(request.state, "user", None)
    
    if user and hasattr(user, "id"):
        return f"user:{user.id}"
    
    # Fall back to IP address
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host}"


# Common rate limit dependencies
api_rate_limit = rate_limit(requests_per_minute=120)
auth_rate_limit = rate_limit(requests_per_minute=30)
analysis_rate_limit = rate_limit(requests_per_minute=20)
