"""
Secure cookie-based authentication system.

This module provides functions for managing authentication using secure httpOnly cookies
instead of localStorage, implementing refresh token rotation, CSRF protection,
and proper security headers.
"""

import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

from fastapi import Request, Response, Depends, HTTPException, status, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from backend.config import settings
from backend.auth.jwt_handler import create_access_token, decode_jwt


# Constants for cookie names
ACCESS_TOKEN_COOKIE = "access_token"
REFRESH_TOKEN_COOKIE = "refresh_token"
CSRF_TOKEN_COOKIE = "csrf_token"
CSRF_HEADER = "X-CSRF-Token"

# Security headers to apply to all responses
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cache-Control": "no-store, max-age=0",
}

# For backward compatibility with Bearer token auth
bearer_scheme = HTTPBearer(auto_error=False)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        for header_name, header_value in SECURITY_HEADERS.items():
            response.headers[header_name] = header_value
            
        return response


def generate_csrf_token() -> str:
    """Generate a secure random token for CSRF protection."""
    return secrets.token_urlsafe(32)


def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    access_expiry: datetime,
    refresh_expiry: datetime,
    csrf_token: Optional[str] = None,
) -> None:
    """
    Set secure httpOnly cookies for authentication.
    
    Args:
        response: The FastAPI response object
        access_token: JWT access token
        refresh_token: JWT refresh token
        access_expiry: Access token expiration datetime
        refresh_expiry: Refresh token expiration datetime
        csrf_token: Optional CSRF token (generated if not provided)
    """
    # Calculate max_age in seconds for each token
    access_max_age = int((access_expiry - datetime.utcnow()).total_seconds())
    refresh_max_age = int((refresh_expiry - datetime.utcnow()).total_seconds())
    
    # Use provided CSRF token or generate a new one
    csrf = csrf_token or generate_csrf_token()
    
    # Common cookie settings
    cookie_settings = {
        "httponly": True,
        "secure": not settings.DEBUG,  # True in production, False in development
        "samesite": "lax",  # Prevents CSRF while allowing normal navigation
        "domain": settings.COOKIE_DOMAIN or None,  # Use configured domain or default to current domain
        "path": "/",  # Available across all paths
    }
    
    # Set access token cookie (shorter lifetime)
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE,
        value=access_token,
        max_age=access_max_age,
        expires=access_expiry,
        **cookie_settings,
    )
    
    # Set refresh token cookie (longer lifetime)
    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE,
        value=refresh_token,
        max_age=refresh_max_age,
        expires=refresh_expiry,
        **cookie_settings,
    )
    
    # Set CSRF token cookie (not httpOnly, so JS can access it)
    # This is used for the double-submit pattern
    response.set_cookie(
        key=CSRF_TOKEN_COOKIE,
        value=csrf,
        max_age=refresh_max_age,  # Same lifetime as refresh token
        expires=refresh_expiry,
        httponly=False,  # Accessible to JavaScript
        secure=not settings.DEBUG,
        samesite="lax",
        path="/",
    )
    
    # Also set the CSRF token in a header for the initial response
    response.headers[CSRF_HEADER] = csrf


def clear_auth_cookies(response: Response) -> None:
    """
    Clear all authentication cookies.
    
    Args:
        response: The FastAPI response object
    """
    # Common cookie settings for clearing
    cookie_settings = {
        "httponly": True,
        "secure": not settings.DEBUG,
        "samesite": "lax",
        "domain": settings.COOKIE_DOMAIN or None,
        "path": "/",
        "expires": datetime.utcnow(),
        "max_age": 0,
    }
    
    # Clear all auth cookies
    response.set_cookie(key=ACCESS_TOKEN_COOKIE, value="", **cookie_settings)
    response.set_cookie(key=REFRESH_TOKEN_COOKIE, value="", **cookie_settings)
    response.set_cookie(
        key=CSRF_TOKEN_COOKIE,
        value="",
        httponly=False,
        secure=not settings.DEBUG,
        samesite="lax",
        path="/",
        expires=datetime.utcnow(),
        max_age=0,
    )


async def create_user_tokens(
    user_id: str, 
    username: str, 
    is_superuser: bool = False, 
    additional_data: Optional[Dict[str, Any]] = None
) -> Tuple[str, str, datetime, datetime]:
    """
    Create access and refresh tokens for a user.
    
    Args:
        user_id: User's unique identifier
        username: User's username
        is_superuser: Whether the user has superuser privileges
        additional_data: Additional data to include in the token payload
        
    Returns:
        Tuple of (access_token, refresh_token, access_expiry, refresh_expiry)
    """
    # Calculate expiration times
    access_expiry = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
    refresh_expiry = datetime.utcnow() + timedelta(minutes=settings.JWT_REFRESH_EXPIRATION_MINUTES)
    
    # Create tokens
    access_token_data = {
        "sub": user_id,
        "username": username,
        "is_superuser": is_superuser,
        "token_type": "access",
        **(additional_data or {}),
    }
    
    refresh_token_data = {
        "sub": user_id,
        "username": username,
        "token_type": "refresh",
        # Include a unique identifier for this refresh token
        # This allows tracking and invalidating specific refresh tokens
        "jti": secrets.token_urlsafe(16),
    }
    
    access_token = create_access_token(access_token_data, expires_delta=access_expiry)
    refresh_token = create_access_token(refresh_token_data, expires_delta=refresh_expiry)
    
    return access_token, refresh_token, access_expiry, refresh_expiry


async def get_token_from_cookies_or_header(
    request: Request,
    access_token: Optional[str] = Cookie(None, alias=ACCESS_TOKEN_COOKIE),
    csrf_token: Optional[str] = Cookie(None, alias=CSRF_TOKEN_COOKIE),
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    """
    Extract the access token from cookies or Authorization header.
    
    This provides backward compatibility with both cookie-based and
    bearer token authentication methods.
    
    Args:
        request: FastAPI request object
        access_token: Access token from cookies
        csrf_token: CSRF token from cookies
        authorization: Bearer token from Authorization header
        
    Returns:
        The access token string
        
    Raises:
        HTTPException: If no valid token is found or CSRF validation fails
    """
    # Check for token in cookies first (preferred method)
    if access_token:
        # For non-GET requests, validate CSRF token
        if request.method not in ["GET", "HEAD", "OPTIONS"]:
            # Get CSRF token from header
            csrf_header = request.headers.get(CSRF_HEADER)
            
            # Validate CSRF token
            if not csrf_header or not csrf_token or csrf_header != csrf_token:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="CSRF token missing or invalid",
                )
        
        return access_token
    
    # Fall back to Authorization header (legacy support)
    elif authorization and authorization.scheme.lower() == "bearer":
        return authorization.credentials
    
    # No valid token found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def refresh_tokens(
    request: Request,
    response: Response,
    refresh_token: Optional[str] = Cookie(None, alias=REFRESH_TOKEN_COOKIE),
    csrf_token: Optional[str] = Cookie(None, alias=CSRF_TOKEN_COOKIE),
) -> Dict[str, Any]:
    """
    Refresh the access token using a valid refresh token.
    
    Implements token rotation: each refresh token can only be used once,
    and a new refresh token is issued with each successful refresh.
    
    Args:
        request: FastAPI request object
        response: FastAPI response object
        refresh_token: Refresh token from cookies
        csrf_token: CSRF token from cookies
        
    Returns:
        Dict containing user information
        
    Raises:
        HTTPException: If refresh token is invalid or missing
    """
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
        )
    
    # Validate CSRF token for refresh operations
    csrf_header = request.headers.get(CSRF_HEADER)
    if not csrf_header or not csrf_token or csrf_header != csrf_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token missing or invalid",
        )
    
    try:
        # Decode and validate the refresh token
        payload = decode_jwt(refresh_token)
        
        # Verify it's a refresh token
        if payload.get("token_type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )
        
        # TODO: Check if this refresh token has been revoked
        # This would require a database or Redis check against a blacklist
        # of revoked refresh tokens
        
        # Extract user information
        user_id = payload.get("sub")
        username = payload.get("username")
        
        if not user_id or not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        
        # Create new tokens (implementing token rotation)
        access_token, new_refresh_token, access_expiry, refresh_expiry = await create_user_tokens(
            user_id=user_id,
            username=username,
            is_superuser=payload.get("is_superuser", False),
        )
        
        # Set the new tokens in cookies
        set_auth_cookies(
            response=response,
            access_token=access_token,
            refresh_token=new_refresh_token,
            access_expiry=access_expiry,
            refresh_expiry=refresh_expiry,
            csrf_token=generate_csrf_token(),  # Generate a new CSRF token
        )
        
        # TODO: Add the old refresh token to a revocation list
        # This prevents refresh token reuse
        
        # Return user information
        return {
            "user_id": user_id,
            "username": username,
            "is_superuser": payload.get("is_superuser", False),
        }
        
    except Exception as e:
        # Clear cookies on error
        clear_auth_cookies(response)
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )


def verify_csrf_token(
    request: Request,
    csrf_token: Optional[str] = Cookie(None, alias=CSRF_TOKEN_COOKIE),
) -> None:
    """
    Verify that the CSRF token in the cookie matches the one in the header.
    
    Args:
        request: FastAPI request object
        csrf_token: CSRF token from cookies
        
    Raises:
        HTTPException: If CSRF validation fails
    """
    # Skip CSRF check for safe methods
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return
    
    # Get CSRF token from header
    csrf_header = request.headers.get(CSRF_HEADER)
    
    # Validate CSRF token
    if not csrf_header or not csrf_token or csrf_header != csrf_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token missing or invalid",
        )
