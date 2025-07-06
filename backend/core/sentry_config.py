"""
Sentry configuration module for error tracking and performance monitoring.

This module initializes and configures Sentry SDK for the application,
with different settings for development and production environments.
"""

import logging
import os
from typing import Dict, Any, Optional

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _is_sentry_enabled() -> bool:
    """
    Return True if a Sentry client is currently configured for this process.
    """
    return sentry_sdk.Hub.current.client is not None


def init_sentry() -> None:
    """
    Initialize Sentry SDK with appropriate configuration based on environment.
    
    Reads DSN from environment variables and configures Sentry with
    appropriate integrations, sampling rates, and environment settings.
    """
    dsn = os.getenv("SENTRY_DSN")
    
    # Early return if Sentry is not configured
    if not dsn:
        logger.warning(
            "SENTRY_DSN environment variable not set. Sentry error tracking disabled."
        )
        return
    
    env = os.getenv("ENVIRONMENT", "development")
    is_prod = env == "production"
    
    # Configure logging integration
    logging_integration = LoggingIntegration(
        level=logging.INFO,        # Capture info and above as breadcrumbs
        event_level=logging.ERROR  # Send errors as events
    )
    
    # Configure performance sampling
    traces_sample_rate = 1.0 if not is_prod else 0.2
    profiles_sample_rate = 1.0 if not is_prod else 0.1
    
    # Initialize Sentry with all required integrations
    sentry_sdk.init(
        dsn=dsn,
        environment=env,
        release=os.getenv("APP_VERSION", "1.8.0-beta"),
        
        # Integrations
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            SqlalchemyIntegration(),
            RedisIntegration(),
            logging_integration,
        ],
        
        # Performance monitoring
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,
        
        # Additional configuration
        send_default_pii=False,
        max_breadcrumbs=50,
        debug=not is_prod,
        
        # Context tagging
        _experiments={
            "profiles_sample_rate": profiles_sample_rate,
        },
        
        # Error filtering
        before_send=before_send,
    )
    
    logger.info(f"Sentry initialized in {env} mode with DSN: {dsn[:10]}...{dsn[-4:]}")


def before_send(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter and modify events before they are sent to Sentry.
    
    Args:
        event: The event dictionary
        hint: A dictionary containing additional information about the event
        
    Returns:
        Modified event or None if the event should be discarded
    """
    # Filter out specific errors we don't want to track
    if "exc_info" in hint:
        exc_type, exc_value, _ = hint["exc_info"]
        
        # Ignore connection errors from external services
        if "ConnectionError" in exc_type.__name__:
            if any(service in str(exc_value) for service in ["redis", "neo4j", "external-api"]):
                return None
    
    # Add custom context to all events
    if "contexts" not in event:
        event["contexts"] = {}
    
    event["contexts"]["service"] = {
        "name": "coding-analyst-droid",
        "version": os.getenv("APP_VERSION", "1.8.0-beta"),
        "tier": os.getenv("SERVICE_TIER", "backend"),
    }
    
    return event


def set_user_context(user_id: str, username: Optional[str] = None) -> None:
    """
    Set user context for Sentry events.
    
    Args:
        user_id: The user's ID
        username: The user's name (optional)
    """
    if _is_sentry_enabled():
        sentry_sdk.set_user({"id": user_id, "username": username})


def set_tag(key: str, value: str) -> None:
    """
    Set a tag for all future events in the current scope.
    
    Args:
        key: Tag name
        value: Tag value
    """
    if _is_sentry_enabled():
        sentry_sdk.set_tag(key, value)


def capture_message(message: str, level: str = "info") -> None:
    """
    Capture a message with Sentry.
    
    Args:
        message: The message to capture
        level: The log level (default: info)
    """
    if _is_sentry_enabled():
        sentry_sdk.capture_message(message, level=level)


def capture_exception(exc: Optional[Exception] = None) -> None:
    """
    Capture an exception with Sentry.
    
    Args:
        exc: The exception to capture (default: current exception)
    """
    if _is_sentry_enabled():
        sentry_sdk.capture_exception(exc)
