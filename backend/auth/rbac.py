"""
Role-Based Access Control (RBAC) for FastAPI endpoints.

This module provides decorators and utilities for implementing role-based
access control on FastAPI endpoints. It works with the existing JWT
authentication system and checks user roles from request.state.user.
"""

import functools
from typing import Callable, List, Optional, Set, Union

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.auth.dependencies import get_current_user


def require_roles(
    roles: Union[str, List[str], Set[str]],
    error_message: Optional[str] = None
) -> Callable:
    """
    Decorator for FastAPI endpoint that requires specific role(s).
    
    Args:
        roles: Single role or list of roles that are allowed to access the endpoint
        error_message: Optional custom error message for forbidden access
        
    Returns:
        Decorator function that can be applied to FastAPI endpoints
    
    Example:
        @app.get("/admin/settings")
        @require_roles(["admin"])
        async def admin_settings():
            return {"settings": "admin only"}
            
        @app.get("/api/v1/reports")
        @require_roles(["admin", "analyst"])
        async def get_reports():
            return {"reports": [...]}
    """
    # Convert single role to list
    if isinstance(roles, str):
        roles = [roles]
    
    # Convert to set for faster lookups
    allowed_roles = set(roles)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request object
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request is None:
                for _, value in kwargs.items():
                    if isinstance(value, Request):
                        request = value
                        break
            
            if request is None:
                raise ValueError("Request object not found in function arguments")
            
            # Check if user exists in request state
            if not hasattr(request.state, "user") or request.state.user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated"
                )
            
            # Get user from request state
            user = request.state.user
            
            # Check if user has required role
            user_role = user.get("role", "").lower()
            if not user_role or user_role not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=error_message or f"Access denied. Required roles: {', '.join(allowed_roles)}"
                )
            
            # User has required role, proceed with the endpoint
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Dependency version for use with FastAPI depends
async def has_roles(
    request: Request,
    roles: Union[str, List[str], Set[str]],
    error_message: Optional[str] = None
) -> bool:
    """
    Dependency function to check if user has required role(s).
    
    Args:
        request: FastAPI request object
        roles: Single role or list of roles that are allowed
        error_message: Optional custom error message
        
    Returns:
        True if user has required role, raises HTTPException otherwise
        
    Example:
        @app.get("/admin/users")
        async def get_users(
            _: bool = Depends(lambda req: has_roles(req, ["admin"]))
        ):
            return {"users": [...]}
    """
    # Convert single role to list
    if isinstance(roles, str):
        roles = [roles]
    
    # Convert to set for faster lookups
    allowed_roles = set(roles)
    
    # Check if user exists in request state
    if not hasattr(request.state, "user") or request.state.user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    # Get user from request state
    user = request.state.user
    
    # Check if user has required role
    user_role = user.get("role", "").lower()
    if not user_role or user_role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=error_message or f"Access denied. Required roles: {', '.join(allowed_roles)}"
        )
    
    return True


# Predefined role constants
class Roles:
    """Constants for common user roles."""
    ADMIN = "admin"
    ANALYST = "analyst"
    USER = "user"
    COMPLIANCE = "compliance"
    AUDITOR = "auditor"


# Predefined role combinations
class RoleSets:
    """Common combinations of roles for reuse."""
    ADMIN_ONLY = {Roles.ADMIN}
    ANALYSTS_AND_ADMIN = {Roles.ANALYST, Roles.ADMIN}
    COMPLIANCE_TEAM = {Roles.COMPLIANCE, Roles.ADMIN}
    ALL_STAFF = {Roles.ADMIN, Roles.ANALYST, Roles.COMPLIANCE, Roles.AUDITOR}
