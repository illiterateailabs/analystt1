"""
Provider Registry Management

This module is responsible for loading, parsing, and providing access to the 
configurations of all external providers defined in `registry.yaml`.

It handles:
- Loading the YAML configuration file.
- Substituting environment variable placeholders (e.g., `${API_KEY}`).
- Caching the loaded configuration to prevent redundant file I/O.
- Providing simple functions to access provider configurations.
"""

import functools
import logging
import os
import re
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# The path to the provider registry file, relative to this file's location.
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.yaml")

# Regex to find environment variable placeholders like `${VAR_NAME}`.
ENV_VAR_PATTERN = re.compile(r"\$\{(?P<name>[A-Z0-9_]+)\}")


def _substitute_env_vars(config: Any) -> Any:
    """
    Recursively substitutes environment variable placeholders in the configuration.

    Args:
        config: The configuration data (can be a dict, list, or primitive).

    Returns:
        The configuration with placeholders replaced by environment variable values.
    """
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    if isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    if isinstance(config, str):
        match = ENV_VAR_PATTERN.search(config)
        if match:
            var_name = match.group("name")
            value = os.getenv(var_name)
            if value is None:
                logger.warning(
                    f"Environment variable '{var_name}' not set, but referenced in provider registry."
                )
                # Replace with an empty string to avoid leaving the placeholder
                return config.replace(match.group(0), "")
            return config.replace(match.group(0), value)
    return config


@functools.lru_cache(maxsize=1)
def _load_providers() -> List[Dict[str, Any]]:
    """
    Loads, parses, and caches the provider configurations from the YAML file.
    
    This function is cached to ensure it runs only once per application lifecycle.

    Returns:
        A list of provider configuration dictionaries.
        
    Raises:
        FileNotFoundError: If the registry.yaml file cannot be found.
        ValueError: If the YAML is malformed or doesn't contain a 'providers' key.
    """
    logger.info(f"Loading provider registry from: {REGISTRY_PATH}")
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        if not isinstance(raw_config, dict) or "providers" not in raw_config:
            raise ValueError("Provider registry YAML must be a dictionary with a 'providers' key.")

        providers_list = raw_config["providers"]
        if not isinstance(providers_list, list):
            raise ValueError("The 'providers' key must contain a list of provider configurations.")

        # Substitute environment variables
        substituted_providers = _substitute_env_vars(providers_list)
        
        logger.info(f"Successfully loaded and processed {len(substituted_providers)} providers.")
        return substituted_providers

    except FileNotFoundError:
        logger.error(f"Provider registry file not found at '{REGISTRY_PATH}'.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing provider registry YAML: {e}")
        raise ValueError(f"Invalid YAML format in '{REGISTRY_PATH}'") from e


def get_all_providers() -> List[Dict[str, Any]]:
    """
    Retrieves the full list of all configured providers.

    Returns:
        A list of provider configuration dictionaries.
    """
    try:
        return _load_providers()
    except (FileNotFoundError, ValueError):
        # Return an empty list if loading fails to allow graceful degradation
        return []


def get_provider(provider_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the configuration for a specific provider by its ID.

    Args:
        provider_id: The unique identifier of the provider (e.g., "gemini", "neo4j").

    Returns:
        A dictionary containing the provider's configuration, or None if not found.
    """
    all_providers = get_all_providers()
    for provider in all_providers:
        if provider.get("id") == provider_id:
            return provider
    logger.warning(f"Provider with ID '{provider_id}' not found in the registry.")
    return None
