"""
Authentication API endpoints for user registration, login, and token management.

This module provides FastAPI routes for:
- User registration (POST /register)
- User login and JWT token generation (POST /login)
- JWT token refreshing (POST /refresh)
- User logout (POST /logout)
- Retrieving current user information (GET /me)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import aioredis

from backend.database import get_db
from backend.models.user import User
from backend.auth.jwt_handler import create_access_token, decode_token
from backend.auth.dependencies import get_current_user
from backend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# Redis client for token blacklist
redis_client: Optional[aioredis.Redis] = None

async def get_redis():
    global redis_client
    if not redis_client:
        redis_client = await aioredis.create_redis_pool(
            settings.REDIS_URL or 'redis://redis:6379',
            encoding='utf-8'
        )
    return redis_client


# Pydantic models for request/response validation
class UserCreate(BaseModel):
    """Request model for user registration."""
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, description="User's password")
    name: str = Field(..., description="User's full name")
    role: str = Field("analyst", description="User's role (admin, analyst, compliance)")


class UserResponse(BaseModel):
    """Response model for user information."""
    id: str
    email: str
    name: str
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # Enable ORM mode for Pydantic


class TokenResponse(BaseModel):
    """Response model for authentication tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Seconds until token expires


class RefreshRequest(BaseModel):
    """Request model for token refresh."""
    refresh_token: str


class LogoutRequest(BaseModel):
    """Request model for logout."""
    refresh_token: Optional[str] = None


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account"
)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account.
    
    Creates a new user with the provided email, password, name, and role.
    The password is hashed before storage.
    """
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists"
        )
    
    # Create new user with hashed password
    hashed_password = User.hash_password(user_data.password)
    
    new_user = User(
        id=uuid.uuid4(),
        email=user_data.email,
        hashed_password=hashed_password,
        name=user_data.name,
        role=user_data.role.lower(),
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    logger.info(f"New user registered: {new_user.email} with role {new_user.role}")
    return new_user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and get access tokens"
)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate a user and return JWT tokens.
    
    Verifies the user's credentials and returns an access token and refresh token
    if authentication is successful.
    """
    # Find user by email
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    
    # Check if user exists and password is correct
    if not user or not User.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate access token
    access_token_expires = timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.email,
            "role": user.role,
            "name": user.name,
            "id": str(user.id),
            "type": "access"
        },
        expires_delta=access_token_expires
    )
    
    # Generate refresh token with longer expiration
    refresh_token_expires = timedelta(minutes=settings.JWT_REFRESH_EXPIRATION_MINUTES)
    refresh_token = create_access_token(
        data={
            "sub": user.email,
            "id": str(user.id),
            "type": "refresh"
        },
        expires_delta=refresh_token_expires
    )
    
    logger.info(f"User logged in: {user.email}")
    
    # Return tokens
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.JWT_EXPIRATION_MINUTES * 60  # Convert to seconds
    }


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token"
)
async def refresh_token(
    refresh_request: RefreshRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh an access token using a refresh token.
    
    Validates the refresh token and returns a new access token and refresh token
    if the refresh token is valid.
    """
    refresh_token = refresh_request.refresh_token
    
    # Check if token is blacklisted
    redis = await get_redis()
    is_blacklisted = await redis.exists(f"blacklist:{refresh_token}")
    
    if is_blacklisted:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Decode and validate the refresh token
        payload = decode_token(refresh_token)
        
        # Check if token is a refresh token
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from token
        email = payload.get("sub")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Find user in database
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Generate new access token
        access_token_expires = timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
        access_token = create_access_token(
            data={
                "sub": user.email,
                "role": user.role,
                "name": user.name,
                "id": str(user.id),
                "type": "access"
            },
            expires_delta=access_token_expires
        )
        
        # Generate new refresh token
        refresh_token_expires = timedelta(minutes=settings.JWT_REFRESH_EXPIRATION_MINUTES)
        new_refresh_token = create_access_token(
            data={
                "sub": user.email,
                "id": str(user.id),
                "type": "refresh"
            },
            expires_delta=refresh_token_expires
        )
        
        logger.info(f"Token refreshed for user: {user.email}")
        
        # Return new tokens
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": settings.JWT_EXPIRATION_MINUTES * 60  # Convert to seconds
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout and invalidate tokens"
)
async def logout(
    logout_request: LogoutRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Logout a user by invalidating their tokens.
    
    Adds the refresh token to a blacklist to prevent it from being used again.
    Uses Redis for persistent token blacklisting.
    """
    # Add refresh token to blacklist if provided
    if logout_request.refresh_token:
        redis = await get_redis()
        # Set expiration to match JWT refresh token expiration (or slightly longer)
        # This prevents the blacklist from growing indefinitely
        await redis.setex(
            f"blacklist:{logout_request.refresh_token}", 
            int(settings.JWT_REFRESH_EXPIRATION_MINUTES * 60),  # Convert to seconds
            "1"
        )
    
    logger.info(f"User logged out: {current_user.get('sub')}")
    
    return {"message": "Successfully logged out"}


@router.get(
    "/me",
    response_model=Dict[str, Any],
    summary="Get current user information"
)
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get information about the currently authenticated user.
    
    Returns user details based on the JWT token.
    """
    # Get user from database for most up-to-date information
    result = await db.execute(select(User).where(User.email == current_user.get("sub")))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "is_active": user.is_active
    }


@router.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection on application shutdown."""
    global redis_client
    if redis_client is not None:
        redis_client.close()
        await redis_client.wait_closed()
        redis_client = None
        logger.info("Redis connection closed")
