"""
Provider Registry System

This module provides a centralized registry for all external data providers
used in the application, including blockchain data providers, AI services,
execution environments, and databases.

The registry is loaded from a YAML configuration file and provides
typed access to provider configurations with environment variable substitution.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union, cast

import yaml

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_REGISTRY_PATH = Path(__file__).parent / "registry.yaml"

# Type definitions
class AuthConfig(TypedDict, total=False):
    type: str
    header_name: Optional[str]
    query_param: Optional[str]
    key_env_var: Optional[str]
    key_prefix: Optional[str]
    username_env_var: Optional[str]
    password_env_var: Optional[str]
    optional: Optional[bool]

class RetryConfig(TypedDict, total=False):
    max_attempts: int
    initial_backoff_seconds: float
    max_backoff_seconds: float
    backoff_factor: float
    retry_on_status_codes: List[int]

class TimeoutConfig(TypedDict, total=False):
    connect_seconds: float
    read_seconds: float
    total_seconds: float

class RateLimitConfig(TypedDict, total=False):
    requests_per_second: Optional[int]
    requests_per_minute: Optional[int]
    requests_per_day: Optional[int]
    tokens_per_minute: Optional[int]
    compute_units_per_day: Optional[int]
    concurrent_sandboxes: Optional[int]
    sandbox_time_limit_seconds: Optional[int]
    queries_per_day: Optional[int]

class MonitoringConfig(TypedDict, total=False):
    track_latency: bool
    track_errors: bool
    track_usage: bool
    track_query_time: Optional[bool]
    track_connection_pool: Optional[bool]
    track_hit_rate: Optional[bool]
    track_memory_usage: Optional[bool]

class CostTrackingConfig(TypedDict, total=False):
    enabled: bool
    input_token_cost: Optional[float]
    output_token_cost: Optional[float]
    compute_unit_cost: Optional[float]

class ProviderConfig(TypedDict):
    id: str
    type: str
    name: str
    description: str
    base_url: Optional[str]
    connection_uri: Optional[str]
    host: Optional[str]
    port: Optional[int]
    enabled: Optional[bool]
    auth: Optional[AuthConfig]
    retry: Optional[RetryConfig]
    timeout: Optional[TimeoutConfig]
    rate_limits: Optional[RateLimitConfig]
    monitoring: Optional[MonitoringConfig]
    cost_tracking: Optional[CostTrackingConfig]
    endpoints: Optional[Dict[str, str]]
    supported_chains: Optional[List[str]]
    features: Optional[Dict[str, bool]]
    models: Optional[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]]
    environments: Optional[Dict[str, Any]]
    databases: Optional[Dict[str, Any]]
    connection_pool: Optional[Dict[str, Any]]

class ProviderRegistry(TypedDict):
    defaults: Dict[str, Any]
    blockchain: List[ProviderConfig]
    ai: List[ProviderConfig]
    execution: List[ProviderConfig]
    databases: List[ProviderConfig]
    mcp: List[ProviderConfig]

# Module-level registry cache
_registry: Optional[ProviderRegistry] = None

def _substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in configuration values.
    
    Supports the format ${ENV_VAR:-default_value} where default_value is optional.
    
    Args:
        value: The value to process for environment variable substitution
        
    Returns:
        The value with environment variables substituted
    """
    if isinstance(value, str):
        # Match patterns like ${ENV_VAR:-default}
        pattern = r'\${([A-Za-z0-9_]+)(?::-([^}]*))?}'
        
        def replace_env_var(match):
            env_var = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(env_var, default)
        
        return re.sub(pattern, replace_env_var, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    else:
        return value

def load_registry(path: Optional[Union[str, Path]] = None) -> ProviderRegistry:
    """
    Load the provider registry from the YAML configuration file.
    
    Args:
        path: Path to the registry YAML file (optional, uses default if not provided)
        
    Returns:
        The provider registry as a dictionary
        
    Raises:
        FileNotFoundError: If the registry file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
    """
    global _registry
    
    if _registry is not None:
        return _registry
    
    registry_path = Path(path) if path else DEFAULT_REGISTRY_PATH
    
    try:
        with open(registry_path, "r") as f:
            raw_registry = yaml.safe_load(f)
        
        # Substitute environment variables
        processed_registry = _substitute_env_vars(raw_registry)
        
        # Cache the registry
        _registry = cast(ProviderRegistry, processed_registry)
        
        logger.info(f"Provider registry loaded from {registry_path}")
        logger.debug(f"Registry contains {len(_registry['blockchain'])} blockchain providers, "
                    f"{len(_registry['ai'])} AI providers, "
                    f"{len(_registry['execution'])} execution environments, "
                    f"{len(_registry['databases'])} database providers, "
                    f"{len(_registry['mcp'])} MCP servers")
        
        return _registry
    except FileNotFoundError:
        logger.error(f"Provider registry file not found: {registry_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in provider registry: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading provider registry: {e}")
        raise

def get_provider(provider_id: str) -> Optional[ProviderConfig]:
    """
    Get a provider configuration by ID.
    
    Args:
        provider_id: The provider ID to look up
        
    Returns:
        The provider configuration or None if not found
    """
    registry = load_registry()
    
    # Search all provider categories
    for category in ["blockchain", "ai", "execution", "databases", "mcp"]:
        for provider in registry[category]:
            if provider["id"] == provider_id:
                # Apply defaults if not overridden
                merged_provider = {**registry["defaults"], **provider}
                return cast(ProviderConfig, merged_provider)
    
    logger.warning(f"Provider not found: {provider_id}")
    return None

def get_providers_by_category(category: str) -> List[ProviderConfig]:
    """
    Get all providers in a specific category.
    
    Args:
        category: The provider category (blockchain, ai, execution, databases, mcp)
        
    Returns:
        List of provider configurations in the category
        
    Raises:
        KeyError: If the category doesn't exist
    """
    registry = load_registry()
    
    if category not in registry:
        logger.error(f"Invalid provider category: {category}")
        raise KeyError(f"Invalid provider category: {category}")
    
    # Apply defaults to each provider
    providers = []
    for provider in registry[category]:
        # Skip disabled providers
        if provider.get("enabled") is False:
            continue
        
        # Merge defaults with provider-specific config
        merged_provider = {**registry["defaults"], **provider}
        providers.append(cast(ProviderConfig, merged_provider))
    
    return providers

def get_enabled_providers() -> List[ProviderConfig]:
    """
    Get all enabled providers across all categories.
    
    Returns:
        List of all enabled provider configurations
    """
    registry = load_registry()
    
    providers = []
    for category in ["blockchain", "ai", "execution", "databases", "mcp"]:
        for provider in registry[category]:
            # Skip disabled providers
            if provider.get("enabled") is False:
                continue
            
            # Merge defaults with provider-specific config
            merged_provider = {**registry["defaults"], **provider}
            providers.append(cast(ProviderConfig, merged_provider))
    
    return providers

def reload_registry() -> None:
    """
    Force reload of the provider registry from disk.
    
    This is useful when the registry file has been updated.
    """
    global _registry
    _registry = None
    load_registry()
    logger.info("Provider registry reloaded")
