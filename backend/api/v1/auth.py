"""Authentication API endpoints for the Analyst's Augmentation Agent.

This module provides routes for user authentication, registration, token refresh,
and user profile management.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status, Request, Body
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, validator

from backend.auth.dependencies import (
    get_current_user, 
    get_optional_user,
    require_admin,
    auth_rate_limit,
    UserRole
)
from backend.auth.jwt_handler import JWTHandler

# Setup router
router = APIRouter()

# Get logger
logger = logging.getLogger(__name__)


# ---- Pydantic Models ----

class UserCreate(BaseModel):
    """User registration request model."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=100)
    role: UserRole = Field(default=UserRole.ANALYST)
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: EmailStr
    full_name: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime] = None


class UserUpdate(BaseModel):
    """User update request model."""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None


class PasswordChange(BaseModel):
    """Password change request model."""
    current_password: str
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def password_strength(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    """Token refresh request model."""
    refresh_token: str


# ---- Routes ----

@router.post("/token", response_model=TokenResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    _: bool = Depends(auth_rate_limit)
):
    """
    Authenticate user and issue JWT tokens.
    
    This endpoint follows the OAuth2 password flow standard.
    """
    # In a real application, you would validate against a database
    # This is a simplified example with hardcoded users for demonstration
    
    # Mock user database - replace with actual database in production
    users = {
        "admin@example.com": {
            "id": "1",
            "email": "admin@example.com",
            "full_name": "Admin User",
            "password": "Admin123!",  # In production, store hashed passwords
            "role": UserRole.ADMIN,
            "created_at": datetime.utcnow()
        },
        "analyst@example.com": {
            "id": "2",
            "email": "analyst@example.com",
            "full_name": "Analyst User",
            "password": "Analyst123!",
            "role": UserRole.ANALYST,
            "created_at": datetime.utcnow()
        }
    }
    
    # Check if user exists and password is correct
    user = users.get(form_data.username)
    if not user or user["password"] != form_data.password:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create user data for token
    user_data = {
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"]
    }
    
    # Generate tokens
    access_token = JWTHandler.create_access_token(
        subject=user["id"],
        user_data=user_data
    )
    
    refresh_token = JWTHandler.create_refresh_token(
        subject=user["id"]
    )
    
    logger.info(f"User logged in: {user['email']}")
    
    # Return tokens
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 60 * 60  # 1 hour in seconds
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    refresh_data: RefreshRequest,
    _: bool = Depends(auth_rate_limit)
):
    """
    Refresh access token using a valid refresh token.
    """
    try:
        # Generate new tokens
        access_token, refresh_token = JWTHandler.refresh_tokens(
            refresh_token=refresh_data.refresh_token
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 60 * 60  # 1 hour in seconds
        }
        
    except HTTPException as e:
        # Re-raise the exception from the JWT handler
        raise e
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error refreshing token"
        )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    request: Request,
    user_data: UserCreate,
    current_user: Optional[Dict] = Depends(get_optional_user),
    _: bool = Depends(auth_rate_limit)
):
    """
    Register a new user.
    
    Admin role can only be assigned by an existing admin.
    """
    # In a real application, you would store in a database
    # This is a simplified example for demonstration
    
    # Check if trying to create admin user
    if user_data.role == UserRole.ADMIN:
        # Only admins can create other admins
        if not current_user or current_user.get("role") != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can create admin users"
            )
    
    # Mock user creation - replace with database in production
    new_user = {
        "id": "3",  # In production, generate a unique ID
        "email": user_data.email,
        "full_name": user_data.full_name,
        "role": user_data.role,
        "created_at": datetime.utcnow(),
        "last_login": None
    }
    
    logger.info(f"User registered: {new_user['email']}")
    
    return new_user


@router.post("/logout")
async def logout(
    request: Request,
    current_user: Dict = Depends(get_current_user)
):
    """
    Logout current user.
    
    Note: JWT tokens cannot be invalidated without a token blacklist.
    This endpoint is mostly for client-side cleanup.
    """
    # In a production system, you might add the token to a blacklist
    # stored in Redis or another fast database
    
    logger.info(f"User logged out: {current_user.get('email', current_user.get('id'))}")
    
    return {"detail": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_user_profile(
    request: Request,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get current user profile.
    """
    # In a real application, you would fetch from a database
    # This is a simplified example with mock data
    
    # Mock user data - replace with database lookup in production
    user = {
        "id": current_user["id"],
        "email": current_user.get("email", "user@example.com"),
        "full_name": current_user.get("full_name", "User"),
        "role": current_user.get("role", UserRole.ANALYST),
        "created_at": datetime.utcnow(),  # In production, fetch from DB
        "last_login": datetime.utcnow()  # In production, fetch from DB
    }
    
    return user


@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    request: Request,
    user_update: UserUpdate,
    current_user: Dict = Depends(get_current_user)
):
    """
    Update current user profile.
    """
    # In a real application, you would update in a database
    # This is a simplified example with mock data
    
    # Mock updated user - replace with database update in production
    updated_user = {
        "id": current_user["id"],
        "email": user_update.email or current_user.get("email", "user@example.com"),
        "full_name": user_update.full_name or current_user.get("full_name", "User"),
        "role": current_user.get("role", UserRole.ANALYST),
        "created_at": datetime.utcnow(),  # In production, fetch from DB
        "last_login": datetime.utcnow()  # In production, fetch from DB
    }
    
    logger.info(f"User profile updated: {updated_user['email']}")
    
    return updated_user


@router.post("/change-password")
async def change_password(
    request: Request,
    password_data: PasswordChange,
    current_user: Dict = Depends(get_current_user),
    _: bool = Depends(auth_rate_limit)
):
    """
    Change user password.
    """
    # In a real application, you would validate against stored hash
    # and update in a database. This is a simplified example.
    
    # Mock password validation - replace with actual validation in production
    # Pretend current password is correct
    
    logger.info(f"Password changed for user: {current_user.get('email', current_user.get('id'))}")
    
    return {"detail": "Password successfully changed"}


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    request: Request,
    current_user: Dict = Depends(require_admin)
):
    """
    List all users (admin only).
    """
    # In a real application, you would fetch from a database
    # This is a simplified example with mock data
    
    # Mock user list - replace with database query in production
    users = [
        {
            "id": "1",
            "email": "admin@example.com",
            "full_name": "Admin User",
            "role": UserRole.ADMIN,
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow()
        },
        {
            "id": "2",
            "email": "analyst@example.com",
            "full_name": "Analyst User",
            "role": UserRole.ANALYST,
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow()
        }
    ]
    
    return users
