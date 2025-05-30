"""JWT token handling for the Analyst's Augmentation Agent.

This module provides functionality for creating, validating, and refreshing JWT tokens
for authentication and authorization purposes.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple

from fastapi import HTTPException, status
from jose import jwt, JWTError
from pydantic import ValidationError

from backend.config import settings


class JWTHandler:
    """Handler for JWT token operations."""

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
            expires_minutes = settings.jwt_expiration_minutes

        expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
        
        to_encode = {
            "sub": str(subject),
            "exp": expire,
            "iat": datetime.utcnow(),
            "nbf": datetime.utcnow(),
            "aud": settings.jwt_audience,
            "iss": settings.jwt_issuer,
            "jti": f"{subject}_{int(time.time())}"
        }
        
        # Add user data if provided
        if user_data:
            to_encode.update({"user_data": user_data})
            
        # Create token
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.secret_key, 
            algorithm=settings.jwt_algorithm
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
            "aud": f"{settings.jwt_audience}:refresh",
            "iss": settings.jwt_issuer,
            "jti": f"refresh_{subject}_{int(time.time())}",
            "type": "refresh"
        }
            
        # Create token
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.secret_key, 
            algorithm=settings.jwt_algorithm
        )
        
        return encoded_jwt

    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT token to decode

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.jwt_algorithm],
                audience=settings.jwt_audience,
                issuer=settings.jwt_issuer,
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

    @staticmethod
    def decode_refresh_token(token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT refresh token.

        Args:
            token: JWT refresh token to decode

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid, expired, or not a refresh token
        """
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.jwt_algorithm],
                audience=f"{settings.jwt_audience}:refresh",
                issuer=settings.jwt_issuer,
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

    @staticmethod
    def refresh_tokens(refresh_token: str) -> Tuple[str, str]:
        """
        Generate new access and refresh tokens using a valid refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            Tuple of (new_access_token, new_refresh_token)

        Raises:
            HTTPException: If refresh token is invalid or expired
        """
        # Decode and validate the refresh token
        payload = JWTHandler.decode_refresh_token(refresh_token)
        
        # Extract subject from payload
        subject = payload.get("sub")
        if not subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Create new tokens
        new_access_token = JWTHandler.create_access_token(subject=subject)
        new_refresh_token = JWTHandler.create_refresh_token(subject=subject)
        
        return new_access_token, new_refresh_token

    @staticmethod
    def verify_token_and_get_subject(token: str) -> str:
        """
        Verify token and extract subject.

        Args:
            token: JWT token to verify

        Returns:
            Subject from token (usually user ID)

        Raises:
            HTTPException: If token is invalid or expired
        """
        payload = JWTHandler.decode_token(token)
        
        subject = payload.get("sub")
        if not subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return subject
