"""
Authentication API endpoints.

This module provides endpoints for user registration, login, logout,
token refresh, and user information retrieval.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Response, Request, Cookie
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.auth.jwt_handler import create_access_token, decode_jwt, get_password_hash, verify_password
from backend.auth.secure_cookies import (
    set_auth_cookies, 
    clear_auth_cookies, 
    create_user_tokens, 
    refresh_tokens,
    get_token_from_cookies_or_header,
    verify_csrf_token,
    ACCESS_TOKEN_COOKIE,
    REFRESH_TOKEN_COOKIE,
    CSRF_TOKEN_COOKIE,
    CSRF_HEADER,
)
from backend.auth.dependencies import get_current_user
from backend.database import get_db
from backend.models.user import User
from backend.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# OAuth2 scheme for backward compatibility
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


# Request/Response Models
from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """User registration request model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    """User response model."""
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_superuser: bool


class TokenResponse(BaseModel):
    """Token response model for backward compatibility."""
    access_token: str
    token_type: str
    user: UserResponse


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str
    remember_me: bool = False


class RefreshResponse(BaseModel):
    """Refresh token response model."""
    user: UserResponse


class MessageResponse(BaseModel):
    """Generic message response model."""
    message: str


# Helper functions
async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
    """
    Authenticate a user by username and password.
    
    Args:
        db: Database session
        username: Username
        password: Password
        
    Returns:
        User object if authentication is successful, None otherwise
    """
    try:
        # Query user by username
        stmt = select(User).where(User.username == username)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        # Check if user exists and password is correct
        if user and user.verify_password(password):
            return user
        
        return None
    except Exception as e:
        logger.error(f"Error authenticating user: {e}")
        return None


# Auth endpoints
@router.post("/register", response_model=TokenResponse)
async def register_user(
    user_data: UserCreate,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user.
    
    Args:
        user_data: User registration data
        response: FastAPI response object
        db: Database session
        
    Returns:
        JWT token and user information
        
    Raises:
        HTTPException: If username or email already exists
    """
    try:
        # Check if username already exists
        stmt = select(User).where(User.username == user_data.username)
        result = await db.execute(stmt)
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists
        stmt = select(User).where(User.email == user_data.email)
        result = await db.execute(stmt)
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name
        )
        new_user.set_password(user_data.password)
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        # Create access and refresh tokens
        access_token, refresh_token, access_expiry, refresh_expiry = await create_user_tokens(
            user_id=str(new_user.id),
            username=new_user.username,
            is_superuser=new_user.is_superuser
        )
        
        # Set secure cookies
        set_auth_cookies(
            response=response,
            access_token=access_token,
            refresh_token=refresh_token,
            access_expiry=access_expiry,
            refresh_expiry=refresh_expiry
        )
        
        # For backward compatibility, also return token in response body
        user_response = UserResponse(
            id=str(new_user.id),
            username=new_user.username,
            email=new_user.email,
            full_name=new_user.full_name,
            is_active=new_user.is_active,
            is_superuser=new_user.is_superuser
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error registering user"
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate a user and return a JWT token.
    
    Args:
        form_data: Login form data
        response: FastAPI response object
        db: Database session
        
    Returns:
        JWT token and user information
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Authenticate user
        user = await authenticate_user(db, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Set token expiration based on remember_me flag
        expiration_minutes = settings.JWT_EXPIRATION_MINUTES
        refresh_expiration_minutes = settings.JWT_REFRESH_EXPIRATION_MINUTES
        
        if form_data.remember_me:
            # Extend token lifetime for "remember me" option
            expiration_minutes *= 4  # e.g., 4 hours instead of 1 hour
            refresh_expiration_minutes *= 2  # e.g., 14 days instead of 7 days
        
        # Create access and refresh tokens
        access_token, refresh_token, access_expiry, refresh_expiry = await create_user_tokens(
            user_id=str(user.id),
            username=user.username,
            is_superuser=user.is_superuser,
            additional_data={"remember_me": form_data.remember_me}
        )
        
        # Set secure cookies
        set_auth_cookies(
            response=response,
            access_token=access_token,
            refresh_token=refresh_token,
            access_expiry=access_expiry,
            refresh_expiry=refresh_expiry
        )
        
        # For backward compatibility, also return token in response body
        user_response = UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_superuser=user.is_superuser
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging in: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during login"
        )


@router.post("/refresh", response_model=RefreshResponse)
async def refresh(
    request: Request,
    response: Response,
):
    """
    Refresh the access token using a valid refresh token.
    
    Args:
        request: FastAPI request object
        response: FastAPI response object
        
    Returns:
        New access token and user information
        
    Raises:
        HTTPException: If refresh token is invalid
    """
    try:
        # Refresh tokens and get user information
        user_info = await refresh_tokens(request, response)
        
        # Return user information
        user_response = UserResponse(
            id=user_info["user_id"],
            username=user_info["username"],
            email="",  # Email not available in token payload
            is_active=True,  # Assumed active since token is valid
            is_superuser=user_info.get("is_superuser", False)
        )
        
        return RefreshResponse(user=user_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error refreshing token"
        )


@router.post("/logout", response_model=MessageResponse)
async def logout(response: Response):
    """
    Log out a user by clearing authentication cookies.
    
    Args:
        response: FastAPI response object
        
    Returns:
        Success message
    """
    # Clear authentication cookies
    clear_auth_cookies(response)
    
    return MessageResponse(message="Successfully logged out")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get information about the currently authenticated user.
    
    Args:
        current_user: Current user from token
        
    Returns:
        User information
    """
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser
    )


@router.get("/csrf-token")
async def get_csrf_token(
    response: Response,
    csrf_token: Optional[str] = Cookie(None, alias=CSRF_TOKEN_COOKIE)
):
    """
    Get a new CSRF token. This endpoint is useful for SPA applications
    that need to refresh their CSRF token.
    
    Args:
        response: FastAPI response object
        csrf_token: Existing CSRF token from cookies
        
    Returns:
        New CSRF token
    """
    from backend.auth.secure_cookies import generate_csrf_token
    
    # Generate a new CSRF token
    new_csrf_token = generate_csrf_token()
    
    # Set the new CSRF token in a cookie
    response.set_cookie(
        key=CSRF_TOKEN_COOKIE,
        value=new_csrf_token,
        httponly=False,  # Accessible to JavaScript
        secure=not settings.DEBUG,
        samesite="lax",
        path="/",
    )
    
    # Also return it in the response body
    return {"csrf_token": new_csrf_token}
