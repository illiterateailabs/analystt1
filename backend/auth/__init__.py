"""
Authentication and authorization module.

This module provides utilities for JWT authentication, user management,
and role-based access control (RBAC).
"""

from backend.auth.dependencies import get_current_user, RateLimiter
from backend.auth.jwt_handler import create_access_token, decode_token
from backend.auth.rbac import require_roles, has_roles, Roles, RoleSets

__all__ = [
    # Authentication dependencies
    "get_current_user",
    "RateLimiter",
    
    # JWT handlers
    "create_access_token",
    "decode_token",
    
    # RBAC components
    "require_roles",
    "has_roles",
    "Roles",
    "RoleSets"
]
