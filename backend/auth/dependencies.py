"""FastAPI dependencies for authentication and authorization.

This module provides dependencies for JWT token validation, optional authentication,
role-based access control, and rate limiting for FastAPI routes.
"""

import time
from enum import Enum
from typing import Dict, List, Optional, Union, Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2AuthorizationCodeBearer, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from backend.auth.jwt_handler import JWTHandler
from backend.config import settings

# OAuth2 scheme for token extraction from requests
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    auto_error=False  # Don't auto-raise for optional auth
)

# HTTP Bearer scheme (alternative to OAuth2)
http_bearer = HTTPBearer(auto_error=False)


class UserRole(str, Enum):
    """User role enumeration for RBAC."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


# Simple in-memory rate limiting store
# In production, use Redis or similar for distributed rate limiting
_rate_limit_store: Dict[str, Dict[str, Union[int, float]]] = {}


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
    request: Optional[Request] = None
) -> Dict:
    """
    Validate JWT token and return current user.
    
    This dependency extracts and validates the JWT token from either:
    - OAuth2 Authorization header (Bearer token)
    - HTTP Bearer Authorization header
    
    Args:
        token: OAuth2 token from Authorization header
        credentials: HTTP Bearer credentials
        request: FastAPI request object for storing user in state
        
    Returns:
        Dict containing user information from token
        
    Raises:
        HTTPException: If token is missing or invalid
    """
    # Get token from either OAuth2 or HTTP Bearer
    final_token = None
    if token:
        final_token = token
    elif credentials:
        final_token = credentials.credentials
        
    if not final_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Decode and validate token
        payload = JWTHandler.decode_token(final_token)
        
        # Extract user data
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user identifier",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Get additional user data if present
        user_data = payload.get("user_data", {})
        
        # Construct user object
        user = {
            "id": user_id,
            "role": user_data.get("role", UserRole.ANALYST),  # Default to ANALYST
            **user_data
        }
        
        # Store user in request state if request is provided
        if request is not None:
            setattr(request.state, 'user', user)
        
        return user
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    current_user: Optional[Dict] = Depends(get_current_user)
) -> Optional[Dict]:
    """
    Get current user if authenticated, otherwise return None.
    
    This dependency is useful for endpoints that can work with or without authentication.
    
    Args:
        current_user: User from token validation dependency
        
    Returns:
        User dict if authenticated, None otherwise
    """
    return current_user


def require_roles(allowed_roles: List[UserRole]):
    """
    Create a dependency that requires specific roles.
    
    Args:
        allowed_roles: List of roles that are allowed to access the endpoint
        
    Returns:
        Dependency function that validates user roles
    """
    async def role_checker(current_user: Dict = Depends(get_current_user)):
        user_role = current_user.get("role")
        
        # Convert to UserRole enum if string
        if isinstance(user_role, str):
            try:
                user_role = UserRole(user_role)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Invalid role: {user_role}"
                )
        
        # Check if user has required role
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {[role.value for role in allowed_roles]}"
            )
        
        return current_user
    
    return role_checker


# Convenience role dependencies
require_admin = require_roles([UserRole.ADMIN])
require_analyst = require_roles([UserRole.ADMIN, UserRole.ANALYST])
require_viewer = require_roles([UserRole.ADMIN, UserRole.ANALYST, UserRole.VIEWER])


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
    
    if user and user.get("id"):
        return f"user:{user['id']}"
    
    # Fall back to IP address
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host}"


# Common rate limit dependencies
api_rate_limit = rate_limit(requests_per_minute=120)
auth_rate_limit = rate_limit(requests_per_minute=30)
analysis_rate_limit = rate_limit(requests_per_minute=20)
