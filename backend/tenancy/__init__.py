"""
Multi-Tenant Architecture Module - SaaS capabilities for the Analyst Droid platform

This module provides the core infrastructure for multi-tenant isolation, enabling
the platform to securely serve multiple customers with complete data separation.
It implements tenant context propagation, database isolation, and tenant-aware
request handling throughout the application.

Key components:
- TenantContext: Stores and propagates tenant information
- TenantMiddleware: Extracts tenant information from requests
- Database isolation: Strategies for Neo4j and PostgreSQL
- Tenant-aware caching: Prefixing for Redis and other caches
- Tenant management: CRUD operations for tenants

Usage:
    from backend.tenancy import get_tenant_context, TenantRequired
    
    # Get the current tenant context in a route
    @app.get("/protected")
    async def protected_route(tenant: TenantContext = Depends(get_tenant_context)):
        return {"message": f"Hello tenant {tenant.id}"}
    
    # Require a valid tenant for a route
    @app.get("/admin")
    async def admin_route(tenant: TenantContext = Depends(TenantRequired())):
        return {"message": "Admin access for tenant {tenant.id}"}
"""

import os
import json
import logging
import asyncio
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from contextvars import ContextVar
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException, Header, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator

from backend.core.logging import get_logger

# Configure logger
logger = get_logger(__name__)

# Tenant context variable (thread-local storage for async context)
tenant_context_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar("tenant_context", default=None)

# Constants
DEFAULT_TENANT_HEADER = "X-Tenant-ID"
DEFAULT_TENANT = "default"
SYSTEM_TENANT = "system"
HEADER_TENANT_KEY = "header"
JWT_TENANT_KEY = "jwt"


class TenantIsolationLevel(str, Enum):
    """Isolation levels for tenant data"""
    SCHEMA = "schema"          # PostgreSQL schema-based isolation
    DATABASE = "database"      # Separate database for each tenant
    MULTI_DATABASE = "multi_database"  # Neo4j multi-database
    LABEL = "label"            # Neo4j label-based isolation
    FIELD = "field"            # Field-based filtering


class TenantModel(BaseModel):
    """Data model for tenant information"""
    id: str = Field(..., description="Unique tenant identifier")
    name: str = Field(..., description="Display name for the tenant")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    active: bool = Field(True, description="Whether the tenant is active")
    isolation_level: TenantIsolationLevel = Field(
        TenantIsolationLevel.SCHEMA, 
        description="Isolation level for this tenant"
    )
    config: Dict[str, Any] = Field(default_factory=dict, description="Tenant-specific configuration")
    
    class Config:
        orm_mode = True


class TenantContext:
    """
    Context object for tenant information
    
    Stores tenant details and provides methods for tenant-specific operations.
    This object is typically created by the TenantMiddleware and can be
    accessed in route handlers via FastAPI dependencies.
    """
    
    def __init__(
        self,
        tenant_id: str,
        tenant_name: Optional[str] = None,
        isolation_level: Optional[TenantIsolationLevel] = None,
        source: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize tenant context
        
        Args:
            tenant_id: Unique tenant identifier
            tenant_name: Display name for the tenant
            isolation_level: Isolation level for this tenant
            source: Source of the tenant information (header, jwt, etc.)
            config: Tenant-specific configuration
        """
        self.id = tenant_id
        self.name = tenant_name or tenant_id
        self.isolation_level = isolation_level or TenantIsolationLevel.SCHEMA
        self.source = source
        self.config = config or {}
        
        # Set the context variable
        self._token = tenant_context_var.set({
            "id": self.id,
            "name": self.name,
            "isolation_level": self.isolation_level,
            "source": self.source,
            "config": self.config
        })
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - reset the context variable"""
        tenant_context_var.reset(self._token)
    
    @property
    def is_system(self) -> bool:
        """Check if this is the system tenant"""
        return self.id == SYSTEM_TENANT
    
    @property
    def is_default(self) -> bool:
        """Check if this is the default tenant"""
        return self.id == DEFAULT_TENANT
    
    @property
    def pg_schema(self) -> str:
        """Get the PostgreSQL schema name for this tenant"""
        if self.is_system:
            return "public"
        return f"tenant_{self.id}"
    
    @property
    def neo4j_database(self) -> str:
        """Get the Neo4j database name for this tenant"""
        if self.is_system:
            return "neo4j"
        return f"tenant_{self.id}"
    
    @property
    def neo4j_label_prefix(self) -> str:
        """Get the Neo4j label prefix for this tenant"""
        if self.is_system:
            return ""
        return f"T{self.id}_"
    
    @property
    def cache_prefix(self) -> str:
        """Get the cache key prefix for this tenant"""
        if self.is_system:
            return "system:"
        return f"tenant:{self.id}:"
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a tenant-specific configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)


class TenantMiddleware:
    """
    Middleware for extracting tenant information from requests
    
    This middleware extracts tenant information from request headers or JWT tokens
    and sets up the tenant context for the request. It supports multiple sources
    of tenant information with a configurable priority order.
    """
    
    def __init__(
        self,
        tenant_header: str = DEFAULT_TENANT_HEADER,
        default_tenant: str = DEFAULT_TENANT,
        source_priority: List[str] = None,
        get_tenant_from_token: Optional[Callable] = None
    ):
        """
        Initialize tenant middleware
        
        Args:
            tenant_header: HTTP header name for tenant ID
            default_tenant: Default tenant ID if none specified
            source_priority: Priority order for tenant sources
            get_tenant_from_token: Function to extract tenant from JWT token
        """
        self.tenant_header = tenant_header
        self.default_tenant = default_tenant
        self.source_priority = source_priority or [JWT_TENANT_KEY, HEADER_TENANT_KEY]
        self.get_tenant_from_token = get_tenant_from_token
        
        logger.info(f"Initialized TenantMiddleware with header: {tenant_header}")
    
    async def __call__(self, request: Request, call_next):
        """
        Process the request and set up tenant context
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in the chain
            
        Returns:
            Response from the next middleware
        """
        # Extract tenant information
        tenant_info = await self._extract_tenant_info(request)
        
        # Create tenant context
        tenant_id = tenant_info.get("id", self.default_tenant)
        tenant_context = TenantContext(
            tenant_id=tenant_id,
            tenant_name=tenant_info.get("name"),
            isolation_level=tenant_info.get("isolation_level"),
            source=tenant_info.get("source"),
            config=tenant_info.get("config", {})
        )
        
        # Add tenant ID to request state
        request.state.tenant_id = tenant_id
        
        # Process request with tenant context
        try:
            response = await call_next(request)
            
            # Add tenant ID to response headers for debugging
            response.headers[self.tenant_header] = tenant_id
            
            return response
        finally:
            # Reset tenant context
            tenant_context_var.reset(tenant_context._token)
    
    async def _extract_tenant_info(self, request: Request) -> Dict[str, Any]:
        """
        Extract tenant information from the request
        
        Args:
            request: FastAPI request object
            
        Returns:
            Dictionary with tenant information
        """
        tenant_info = {}
        
        # Try each source in priority order
        for source in self.source_priority:
            if source == HEADER_TENANT_KEY:
                # Extract from header
                header_value = request.headers.get(self.tenant_header)
                if header_value:
                    tenant_info = {
                        "id": header_value,
                        "source": HEADER_TENANT_KEY
                    }
                    break
            
            elif source == JWT_TENANT_KEY and self.get_tenant_from_token:
                # Extract from JWT token
                auth_header = request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    token = auth_header.replace("Bearer ", "")
                    try:
                        token_tenant_info = await self.get_tenant_from_token(token)
                        if token_tenant_info and "id" in token_tenant_info:
                            tenant_info = token_tenant_info
                            tenant_info["source"] = JWT_TENANT_KEY
                            break
                    except Exception as e:
                        logger.warning(f"Error extracting tenant from token: {str(e)}")
        
        # Use default tenant if no tenant found
        if not tenant_info:
            tenant_info = {
                "id": self.default_tenant,
                "source": "default"
            }
        
        return tenant_info


def get_tenant_context() -> TenantContext:
    """
    Get the current tenant context
    
    This function is designed to be used as a FastAPI dependency to inject
    the current tenant context into route handlers.
    
    Returns:
        Current tenant context or default context if none set
    """
    context = tenant_context_var.get()
    if context:
        return TenantContext(**context)
    
    # Return default context if none set
    return TenantContext(
        tenant_id=DEFAULT_TENANT,
        source="default"
    )


class TenantRequired:
    """
    Dependency for requiring a valid tenant
    
    This dependency ensures that a valid tenant is present in the request.
    It can be used to protect routes that require tenant isolation.
    """
    
    def __init__(
        self,
        allow_default: bool = False,
        allow_system: bool = False
    ):
        """
        Initialize tenant requirement
        
        Args:
            allow_default: Whether to allow the default tenant
            allow_system: Whether to allow the system tenant
        """
        self.allow_default = allow_default
        self.allow_system = allow_system
    
    def __call__(self, tenant: TenantContext = Depends(get_tenant_context)):
        """
        Check if tenant is valid
        
        Args:
            tenant: Tenant context from dependency injection
            
        Returns:
            Tenant context if valid
            
        Raises:
            HTTPException: If tenant is not valid
        """
        if tenant.id == DEFAULT_TENANT and not self.allow_default:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant-specific access required"
            )
        
        if tenant.id == SYSTEM_TENANT and not self.allow_system:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="System tenant access not allowed"
            )
        
        return tenant


def get_tenant_header(
    x_tenant_id: Optional[str] = Header(None, alias=DEFAULT_TENANT_HEADER)
) -> Optional[str]:
    """
    FastAPI dependency for extracting tenant ID from header
    
    Args:
        x_tenant_id: Tenant ID from header
        
    Returns:
        Tenant ID or None
    """
    return x_tenant_id


def get_tenant_id_from_request(request: Request) -> Optional[str]:
    """
    Extract tenant ID from request state or headers
    
    Args:
        request: FastAPI request object
        
    Returns:
        Tenant ID or None
    """
    # Try request state first (set by middleware)
    tenant_id = getattr(request.state, "tenant_id", None)
    
    # Fall back to header
    if not tenant_id:
        tenant_id = request.headers.get(DEFAULT_TENANT_HEADER)
    
    return tenant_id


def create_tenant_filter(tenant_field: str = "tenant_id") -> Callable:
    """
    Create a filter function for tenant-specific database queries
    
    Args:
        tenant_field: Field name for tenant ID in database
        
    Returns:
        Filter function that adds tenant condition to queries
    """
    def filter_by_tenant(query, tenant_id=None):
        """Add tenant filter to query"""
        if tenant_id is None:
            # Get current tenant from context
            context = tenant_context_var.get()
            if context:
                tenant_id = context.get("id")
        
        if tenant_id and tenant_id != SYSTEM_TENANT:
            # Add tenant filter
            return query.filter_by(**{tenant_field: tenant_id})
        
        return query
    
    return filter_by_tenant


def get_tenant_db_config(tenant: TenantContext) -> Dict[str, Any]:
    """
    Get database configuration for a tenant
    
    Args:
        tenant: Tenant context
        
    Returns:
        Dictionary with database configuration
    """
    isolation_level = tenant.isolation_level
    
    if isolation_level == TenantIsolationLevel.SCHEMA:
        return {
            "schema": tenant.pg_schema,
            "type": "schema"
        }
    elif isolation_level == TenantIsolationLevel.DATABASE:
        return {
            "database": f"tenant_{tenant.id}",
            "type": "database"
        }
    elif isolation_level == TenantIsolationLevel.MULTI_DATABASE:
        return {
            "database": tenant.neo4j_database,
            "type": "multi_database"
        }
    elif isolation_level == TenantIsolationLevel.LABEL:
        return {
            "label_prefix": tenant.neo4j_label_prefix,
            "type": "label"
        }
    elif isolation_level == TenantIsolationLevel.FIELD:
        return {
            "field": "tenant_id",
            "value": tenant.id,
            "type": "field"
        }
    
    # Default
    return {
        "type": "none"
    }


def apply_tenant_context_to_query(query: str, tenant: TenantContext) -> str:
    """
    Apply tenant context to a database query
    
    Args:
        query: Original query string
        tenant: Tenant context
        
    Returns:
        Modified query with tenant isolation
    """
    isolation_level = tenant.isolation_level
    
    if isolation_level == TenantIsolationLevel.SCHEMA:
        # Add schema prefix to table names
        # This is a simplified example - real implementation would use a proper SQL parser
        schema = tenant.pg_schema
        return query.replace("FROM ", f"FROM {schema}.")
    
    elif isolation_level == TenantIsolationLevel.FIELD:
        # Add tenant_id condition to WHERE clause
        if "WHERE" in query:
            return query.replace("WHERE ", f"WHERE tenant_id = '{tenant.id}' AND ")
        elif "GROUP BY" in query:
            return query.replace("GROUP BY", f"WHERE tenant_id = '{tenant.id}' GROUP BY")
        elif "ORDER BY" in query:
            return query.replace("ORDER BY", f"WHERE tenant_id = '{tenant.id}' ORDER BY")
        else:
            return f"{query} WHERE tenant_id = '{tenant.id}'"
    
    # For other isolation levels, no modification needed at query level
    return query


def get_tenant_cache_key(base_key: str, tenant: Optional[TenantContext] = None) -> str:
    """
    Get a tenant-specific cache key
    
    Args:
        base_key: Base cache key
        tenant: Tenant context (or None to use current context)
        
    Returns:
        Tenant-specific cache key
    """
    if tenant is None:
        tenant = get_tenant_context()
    
    return f"{tenant.cache_prefix}{base_key}"


def create_tenant(
    tenant_id: str,
    tenant_name: str,
    isolation_level: TenantIsolationLevel = TenantIsolationLevel.SCHEMA,
    config: Optional[Dict[str, Any]] = None
) -> TenantModel:
    """
    Create a new tenant
    
    Args:
        tenant_id: Unique tenant identifier
        tenant_name: Display name for the tenant
        isolation_level: Isolation level for this tenant
        config: Tenant-specific configuration
        
    Returns:
        Created tenant model
    """
    # This is a placeholder - actual implementation would store in database
    tenant = TenantModel(
        id=tenant_id,
        name=tenant_name,
        isolation_level=isolation_level,
        config=config or {}
    )
    
    logger.info(f"Created tenant: {tenant_id} ({tenant_name})")
    return tenant


# Initialize on module import
logger.info("Tenant module initialized")
