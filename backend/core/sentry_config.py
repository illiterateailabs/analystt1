"""
Sentry configuration for error monitoring and performance tracking.

This module provides a comprehensive configuration for Sentry integration,
including error monitoring, performance tracking, sampling strategies,
and context enrichment for the Analyst Agent application.
"""

import logging
import os
import re
import socket
from typing import Any, Dict, Optional, List, Callable

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.httpx import HttpxIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from backend.config import settings

logger = logging.getLogger(__name__)

# Sensitive data patterns to scrub from error reports
SENSITIVE_KEYS = [
    # Authentication related
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"auth", re.IGNORECASE),
    re.compile(r"key", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    
    # Personal data
    re.compile(r"email", re.IGNORECASE),
    re.compile(r"phone", re.IGNORECASE),
    re.compile(r"address", re.IGNORECASE),
    re.compile(r"name", re.IGNORECASE),
    re.compile(r"ssn", re.IGNORECASE),
    re.compile(r"social", re.IGNORECASE),
    
    # Financial data
    re.compile(r"card", re.IGNORECASE),
    re.compile(r"account", re.IGNORECASE),
    re.compile(r"bank", re.IGNORECASE),
    re.compile(r"payment", re.IGNORECASE),
]

# Error types to ignore (noisy or expected errors)
IGNORED_ERRORS = [
    # Network related
    "ConnectionRefusedError",
    "ConnectionResetError",
    "ConnectionError",
    "TimeoutError",
    
    # Client related
    "ClientDisconnect",
    "ClientConnectionError",
    
    # Expected application errors
    "ValidationError",  # Pydantic validation errors
]


def before_send(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter and modify events before sending to Sentry.
    
    This function:
    1. Drops ignored error types
    2. Scrubs sensitive data
    3. Adds additional context
    
    Args:
        event: Sentry event data
        hint: Additional information about the event
        
    Returns:
        Modified event or None to drop the event
    """
    # Check if error should be ignored
    if "exc_info" in hint:
        exc_type, exc_value, _ = hint["exc_info"]
        if exc_type.__name__ in IGNORED_ERRORS:
            return None
    
    # Scrub sensitive data from request bodies
    if "request" in event and "data" in event["request"]:
        data = event["request"]["data"]
        if isinstance(data, dict):
            for key in list(data.keys()):
                if any(pattern.search(key) for pattern in SENSITIVE_KEYS):
                    data[key] = "[REDACTED]"
    
    # Scrub sensitive data from headers
    if "request" in event and "headers" in event["request"]:
        headers = event["request"]["headers"]
        if isinstance(headers, dict):
            for key in list(headers.keys()):
                if any(pattern.search(key) for pattern in SENSITIVE_KEYS):
                    headers[key] = "[REDACTED]"
    
    # Add additional context
    event.setdefault("tags", {})
    event["tags"]["host"] = socket.gethostname()
    
    return event


def traces_sampler(sampling_context: Dict[str, Any]) -> float:
    """
    Determine sampling rate based on the event context.
    
    This allows for different sampling rates for different types of operations:
    - Health checks and static files have low sampling rates
    - Critical operations have high sampling rates
    - Everything else has a default rate
    
    Args:
        sampling_context: Information about the current span/transaction
        
    Returns:
        Sampling rate between 0.0 and 1.0
    """
    # Default sample rate
    sample_rate = 0.1  # 10% of transactions
    
    # Extract relevant information
    transaction_name = sampling_context.get("transaction_name", "")
    path = sampling_context.get("wsgi_environ", {}).get("PATH_INFO", "")
    
    # Lower sampling for health checks and static files
    if transaction_name.startswith("GET /health") or path.startswith("/health"):
        return 0.01  # 1% sample rate for health checks
    
    if path.startswith("/static/") or path.endswith((".js", ".css", ".png", ".jpg", ".ico")):
        return 0.01  # 1% sample rate for static files
    
    # Higher sampling for critical operations
    if any(critical in path for critical in ["/auth/", "/api/v1/analysis/", "/api/v1/crew/"]):
        return 0.5  # 50% sample rate for critical operations
    
    # Higher sampling for error events
    if "event" in sampling_context:
        # If it's an error event, capture it more frequently
        return 0.8  # 80% sample rate for errors
    
    return sample_rate


def configure_scope_with_user(user_id: str, username: str, is_superuser: bool = False) -> None:
    """
    Configure the Sentry scope with user information.
    
    Call this function when a user is authenticated to associate
    subsequent events with the user.
    
    Args:
        user_id: User's ID
        username: User's username
        is_superuser: Whether the user is a superuser
    """
    sentry_sdk.set_user({
        "id": user_id,
        "username": username,
        "is_superuser": is_superuser
    })


def add_breadcrumb(
    category: str,
    message: str,
    level: str = "info",
    data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add a breadcrumb to the current scope.
    
    Breadcrumbs provide additional context for errors by showing
    what happened leading up to an error.
    
    Args:
        category: Breadcrumb category
        message: Breadcrumb message
        level: Severity level
        data: Additional data
    """
    sentry_sdk.add_breadcrumb(
        category=category,
        message=message,
        level=level,
        data=data
    )


def set_tag(key: str, value: str) -> None:
    """
    Set a tag on the current scope.
    
    Tags are key-value pairs that are indexed and searchable.
    
    Args:
        key: Tag key
        value: Tag value
    """
    sentry_sdk.set_tag(key, value)


def set_context(name: str, context: Dict[str, Any]) -> None:
    """
    Add contextual data to the current scope.
    
    Context provides additional information about the environment
    when an error occurs.
    
    Args:
        name: Context name
        context: Context data
    """
    sentry_sdk.set_context(name, context)


def setup_sentry() -> None:
    """
    Initialize Sentry SDK with all integrations and configuration.
    
    This function should be called during application startup.
    """
    dsn = settings.SENTRY_DSN
    
    if not dsn:
        logger.warning("Sentry DSN not configured. Error tracking disabled.")
        return
    
    # Initialize Sentry with all integrations
    sentry_sdk.init(
        dsn=dsn,
        environment=settings.ENVIRONMENT,
        release=f"analyst-agent@{settings.APP_VERSION}",
        
        # Integrations
        integrations=[
            # FastAPI integration
            FastApiIntegration(transaction_style="endpoint"),
            
            # Database integration
            SqlalchemyIntegration(),
            
            # Redis integration
            RedisIntegration(),
            
            # Logging integration
            LoggingIntegration(
                level=logging.INFO,        # Capture info and above as breadcrumbs
                event_level=logging.ERROR  # Send errors as events
            ),
            
            # Async support
            AsyncioIntegration(),
            
            # HTTP client integration
            HttpxIntegration(),
            
            # Starlette integration (ASGI)
            StarletteIntegration(),
        ],
        
        # Performance monitoring
        traces_sample_rate=0.1,  # Base sample rate, refined by traces_sampler
        traces_sampler=traces_sampler,
        
        # Event filtering
        before_send=before_send,
        
        # Additional configuration
        send_default_pii=False,  # Don't send personally identifiable information
        max_breadcrumbs=50,      # Store more breadcrumbs for better debugging
        debug=settings.DEBUG,     # Enable debug mode in development
        
        # Hooks
        auto_enabling_integrations=True,
        auto_session_tracking=True,
        
        # Performance
        _experiments={
            # Optimize SDK performance
            "profiles_sample_rate": 0.1,  # Enable profiling for 10% of transactions
        },
    )
    
    # Set global tags
    sentry_sdk.set_tag("app_name", settings.APP_NAME)
    sentry_sdk.set_tag("app_version", settings.APP_VERSION)
    
    logger.info(f"Sentry initialized for environment: {settings.ENVIRONMENT}")


def capture_exception(exception: Exception, **kwargs) -> str:
    """
    Capture an exception with additional context.
    
    Args:
        exception: The exception to capture
        **kwargs: Additional context to include
        
    Returns:
        The event ID
    """
    return sentry_sdk.capture_exception(exception, **kwargs)


def capture_message(message: str, level: str = "info", **kwargs) -> str:
    """
    Capture a message with the given level.
    
    Args:
        message: The message to capture
        level: The severity level
        **kwargs: Additional context to include
        
    Returns:
        The event ID
    """
    return sentry_sdk.capture_message(message, level, **kwargs)


def start_transaction(name: str, op: str, **kwargs) -> Any:
    """
    Start a new transaction for performance monitoring.
    
    Args:
        name: Transaction name
        op: Operation type
        **kwargs: Additional transaction parameters
        
    Returns:
        Transaction object
    """
    return sentry_sdk.start_transaction(name=name, op=op, **kwargs)
