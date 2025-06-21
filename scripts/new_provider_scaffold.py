#!/usr/bin/env python3
"""
Provider & Tool Scaffold Generator

This script generates scaffolding for new API providers and tools for the
coding-analyst-droid platform. It creates:
1. Provider registry entry in backend/providers/registry.yaml
2. Client integration file in backend/integrations/
3. Tool classes in backend/agents/tools/
4. Tests for the provider and tools
5. Updates necessary configuration files

Usage:
    python scripts/new_provider_scaffold.py
    python scripts/new_provider_scaffold.py --name "example_provider" --type "rest"
    python scripts/new_provider_scaffold.py --config provider_config.json

Options:
    --name NAME         Provider name (snake_case)
    --type TYPE         Provider type (rest, graphql, websocket)
    --config FILE       Path to JSON configuration file
    --output-dir DIR    Output directory for generated files
    --skip-tests        Skip generating test files
    --force             Overwrite existing files
    --help              Show this help message and exit
"""

import argparse
import json
import os
import re
import sys
import textwrap
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# Default templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Default paths
DEFAULT_PATHS = {
    "provider_registry": Path("backend/providers/registry.yaml"),
    "integrations_dir": Path("backend/integrations"),
    "tools_dir": Path("backend/agents/tools"),
    "tests_dir": Path("tests"),
}

# Provider types
PROVIDER_TYPES = ["rest", "graphql", "websocket", "custom"]

# Authentication methods
AUTH_METHODS = ["api_key", "oauth", "bearer_token", "basic_auth", "none"]

# Data types for providers
DATA_TYPES = [
    "blockchain_transactions",
    "token_transfers",
    "nft_data",
    "defi_protocols",
    "wallet_balances",
    "contract_events",
    "market_data",
    "custom",
]


class ProviderConfig:
    """Configuration for a new provider."""
    
    def __init__(
        self,
        name: str,
        provider_type: str,
        base_url: str,
        auth_method: str,
        data_types: List[str],
        description: str = "",
        version: str = "1.0.0",
        rate_limit: Optional[Dict[str, Any]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        endpoints: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.provider_type = provider_type
        self.base_url = base_url
        self.auth_method = auth_method
        self.data_types = data_types
        self.description = description
        self.version = version
        self.rate_limit = rate_limit or {
            "requests_per_minute": 60,
            "requests_per_day": 10000,
        }
        self.retry_config = retry_config or {
            "max_retries": 3,
            "backoff_factor": 1.0,
            "retry_status_codes": [429, 500, 502, 503, 504],
        }
        self.headers = headers or {}
        self.endpoints = endpoints or {}
        self.tools = tools or []
        self.extra_config = extra_config or {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProviderConfig':
        """Create a provider config from a dictionary."""
        return cls(
            name=data.get("name", ""),
            provider_type=data.get("provider_type", "rest"),
            base_url=data.get("base_url", ""),
            auth_method=data.get("auth_method", "api_key"),
            data_types=data.get("data_types", []),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            rate_limit=data.get("rate_limit"),
            retry_config=data.get("retry_config"),
            headers=data.get("headers"),
            endpoints=data.get("endpoints"),
            tools=data.get("tools"),
            extra_config=data.get("extra_config"),
        )
    
    @classmethod
    def from_json_file(cls, file_path: Union[str, Path]) -> 'ProviderConfig':
        """Load provider config from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert provider config to a dictionary."""
        return {
            "name": self.name,
            "provider_type": self.provider_type,
            "base_url": self.base_url,
            "auth_method": self.auth_method,
            "data_types": self.data_types,
            "description": self.description,
            "version": self.version,
            "rate_limit": self.rate_limit,
            "retry_config": self.retry_config,
            "headers": self.headers,
            "endpoints": self.endpoints,
            "tools": self.tools,
            "extra_config": self.extra_config,
        }
    
    def to_json_file(self, file_path: Union[str, Path]) -> None:
        """Save provider config to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_registry_yaml(self) -> Dict[str, Any]:
        """Convert provider config to registry YAML format."""
        registry_entry = {
            "id": self.name,
            "type": self.provider_type,
            "base_url": self.base_url,
            "description": self.description,
            "version": self.version,
            "auth": {
                "method": self.auth_method,
                "env_var": f"{self.name.upper()}_API_KEY",
            },
            "rate_limit": self.rate_limit,
            "retry": self.retry_config,
        }
        
        # Add headers if present
        if self.headers:
            registry_entry["headers"] = self.headers
        
        # Add endpoints if present
        if self.endpoints:
            registry_entry["endpoints"] = self.endpoints
        
        # Add extra config if present
        if self.extra_config:
            for key, value in self.extra_config.items():
                registry_entry[key] = value
        
        return registry_entry


class ScaffoldGenerator:
    """Generator for provider and tool scaffolding."""
    
    def __init__(
        self,
        config: ProviderConfig,
        paths: Dict[str, Path] = None,
        force: bool = False,
        skip_tests: bool = False,
    ):
        self.config = config
        self.paths = paths or DEFAULT_PATHS
        self.force = force
        self.skip_tests = skip_tests
        
        # Ensure paths exist
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
    
    def generate_all(self) -> Dict[str, Path]:
        """Generate all scaffolding files."""
        generated_files = {}
        
        # Update provider registry
        registry_path = self.update_provider_registry()
        generated_files["provider_registry"] = registry_path
        
        # Generate client integration
        client_path = self.generate_client_integration()
        generated_files["client_integration"] = client_path
        
        # Generate tools
        tool_paths = self.generate_tools()
        generated_files["tools"] = tool_paths
        
        # Generate tests
        if not self.skip_tests:
            test_paths = self.generate_tests()
            generated_files["tests"] = test_paths
        
        return generated_files
    
    def update_provider_registry(self) -> Path:
        """Update provider registry YAML with new provider."""
        registry_path = self.paths["provider_registry"]
        
        # Load existing registry
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                try:
                    registry = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    registry = {}
        else:
            registry = {}
        
        # Add or update provider entry
        providers = registry.get("providers", [])
        
        # Check if provider already exists
        provider_exists = False
        for i, provider in enumerate(providers):
            if provider.get("id") == self.config.name:
                if not self.force:
                    raise ValueError(f"Provider '{self.config.name}' already exists in registry. Use --force to overwrite.")
                # Update existing provider
                providers[i] = self.config.to_registry_yaml()
                provider_exists = True
                break
        
        # Add new provider if it doesn't exist
        if not provider_exists:
            providers.append(self.config.to_registry_yaml())
        
        # Update registry
        registry["providers"] = providers
        
        # Save registry
        with open(registry_path, 'w') as f:
            yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Updated provider registry: {registry_path}")
        return registry_path
    
    def generate_client_integration(self) -> Path:
        """Generate client integration file."""
        client_path = self.paths["integrations_dir"] / f"{self.config.name}_client.py"
        
        # Check if file already exists
        if client_path.exists() and not self.force:
            raise ValueError(f"Client integration file already exists: {client_path}. Use --force to overwrite.")
        
        # Generate client code based on provider type
        if self.config.provider_type == "rest":
            client_code = self._generate_rest_client()
        elif self.config.provider_type == "graphql":
            client_code = self._generate_graphql_client()
        elif self.config.provider_type == "websocket":
            client_code = self._generate_websocket_client()
        else:
            client_code = self._generate_custom_client()
        
        # Write client code to file
        with open(client_path, 'w') as f:
            f.write(client_code)
        
        print(f"✅ Generated client integration: {client_path}")
        return client_path
    
    def generate_tools(self) -> List[Path]:
        """Generate tool classes for the provider."""
        tool_paths = []
        
        # Create tools directory for provider if needed
        provider_tools_dir = self.paths["tools_dir"] / self.config.name
        os.makedirs(provider_tools_dir, exist_ok=True)
        
        # Create __init__.py if it doesn't exist
        init_path = provider_tools_dir / "__init__.py"
        if not init_path.exists():
            with open(init_path, 'w') as f:
                f.write(f'"""\nTools for {self.config.description or self.config.name} integration.\n"""\n')
        
        tool_paths.append(init_path)
        
        # Generate tool classes
        if not self.config.tools:
            # Generate a default tool if none specified
            default_tool = {
                "name": f"{self.config.name}_data_tool",
                "description": f"Tool for retrieving data from {self.config.description or self.config.name}",
                "endpoints": ["get_data"],
                "data_types": self.config.data_types,
            }
            self.config.tools.append(default_tool)
        
        for tool_config in self.config.tools:
            tool_path = self._generate_tool(tool_config, provider_tools_dir)
            tool_paths.append(tool_path)
        
        return tool_paths
    
    def generate_tests(self) -> List[Path]:
        """Generate tests for provider and tools."""
        test_paths = []
        
        # Generate provider client test
        client_test_path = self.paths["tests_dir"] / f"test_{self.config.name}_client.py"
        client_test_code = self._generate_client_test()
        
        with open(client_test_path, 'w') as f:
            f.write(client_test_code)
        
        test_paths.append(client_test_path)
        print(f"✅ Generated client test: {client_test_path}")
        
        # Generate tool tests
        for tool_config in self.config.tools:
            tool_name = tool_config["name"]
            tool_test_path = self.paths["tests_dir"] / f"test_{tool_name}.py"
            tool_test_code = self._generate_tool_test(tool_config)
            
            with open(tool_test_path, 'w') as f:
                f.write(tool_test_code)
            
            test_paths.append(tool_test_path)
            print(f"✅ Generated tool test: {tool_test_path}")
        
        # Generate integration test
        integration_test_path = self.paths["tests_dir"] / f"test_{self.config.name}_integration.py"
        integration_test_code = self._generate_integration_test()
        
        with open(integration_test_path, 'w') as f:
            f.write(integration_test_code)
        
        test_paths.append(integration_test_path)
        print(f"✅ Generated integration test: {integration_test_path}")
        
        return test_paths
    
    def _generate_rest_client(self) -> str:
        """Generate REST client code."""
        class_name = self._to_camel_case(f"{self.config.name}Client")
        
        # Generate endpoint methods
        endpoint_methods = []
        
        # Add default endpoints if none provided
        if not self.config.endpoints:
            self.config.endpoints = {
                "get_data": "/api/v1/data",
                "get_transactions": "/api/v1/transactions",
            }
        
        for endpoint_name, endpoint_path in self.config.endpoints.items():
            method_code = f"""
    async def {endpoint_name}(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieve data from {endpoint_path} endpoint."""
        url = f"{{self.base_url}}{endpoint_path}"
        return await self._make_request("GET", url, params=params)
"""
            endpoint_methods.append(method_code)
        
        # Generate client code
        return f"""\"\"\"
{self.config.name.replace('_', ' ').title()} API Client

This module provides a client for interacting with the {self.config.description or self.config.name} API.
It handles authentication, rate limiting, and request retries.

Generated by new_provider_scaffold.py on {datetime.now().strftime('%Y-%m-%d')}
\"\"\"

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import httpx

from backend.core.metrics import ApiMetrics
from backend.providers import get_provider

# Configure module logger
logger = logging.getLogger(__name__)


class {class_name}:
    \"\"\"Client for {self.config.description or self.config.name} API.\"\"\"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        \"\"\"
        Initialize the {self.config.name} client.
        
        Args:
            api_key: API key for authentication (defaults to environment variable)
            base_url: Base URL for API (defaults to provider registry)
            timeout: Request timeout in seconds
        \"\"\"
        # Get provider configuration
        provider_config = get_provider("{self.config.name}")
        if not provider_config:
            raise ValueError(f"Provider '{self.config.name}' not found in registry")
        
        # Set base URL
        self.base_url = base_url or provider_config.get("base_url", "{self.config.base_url}")
        
        # Set API key
        self.api_key = api_key or os.environ.get(
            "{self.config.name.upper()}_API_KEY",
            provider_config.get("auth", {}).get("key", "")
        )
        
        if not self.api_key and "{self.config.auth_method}" != "none":
            logger.warning(f"No API key provided for {self.config.name}")
        
        # Set timeout
        self.timeout = timeout
        
        # Set up headers
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        
        # Add authentication header if needed
        if self.api_key:
            if "{self.config.auth_method}" == "api_key":
                self.headers["X-API-Key"] = self.api_key
            elif "{self.config.auth_method}" == "bearer_token":
                self.headers["Authorization"] = f"Bearer {{self.api_key}}"
        
        # Add additional headers from provider config
        provider_headers = provider_config.get("headers", {})
        self.headers.update(provider_headers)
        
        # Get rate limit configuration
        self.rate_limit = provider_config.get("rate_limit", {
            "requests_per_minute": {self.config.rate_limit.get('requests_per_minute', 60)},
            "requests_per_day": {self.config.rate_limit.get('requests_per_day', 10000)},
        })
        
        # Get retry configuration
        self.retry_config = provider_config.get("retry", {
            "max_retries": {self.config.retry_config.get('max_retries', 3)},
            "backoff_factor": {self.config.retry_config.get('backoff_factor', 1.0)},
            "retry_status_codes": {self.config.retry_config.get('retry_status_codes', [429, 500, 502, 503, 504])},
        })
        
        logger.info(f"Initialized {self.config.name} client with base URL: {{self.base_url}}")
    
    async def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        \"\"\"
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            params: Query parameters
            data: Request body data
            headers: Additional headers
            
        Returns:
            Response data
            
        Raises:
            httpx.HTTPError: If the request fails
        \"\"\"
        start_time = time.time()
        request_headers = {{**self.headers}}
        if headers:
            request_headers.update(headers)
        
        # Track API call
        endpoint = url.replace(self.base_url, "")
        api_metrics = ApiMetrics.track_call(
            provider="{self.config.name}",
            endpoint=endpoint,
            func=lambda: None,
            environment="development",
            version="1.8.0-beta",
        )
        
        # Make request with retries
        retries = 0
        max_retries = self.retry_config.get("max_retries", 3)
        backoff_factor = self.retry_config.get("backoff_factor", 1.0)
        retry_status_codes = self.retry_config.get("retry_status_codes", [429, 500, 502, 503, 504])
        
        while True:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        params=params,
                        json=data,
                        headers=request_headers,
                    )
                    
                    # Check for rate limiting
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 1))
                        logger.warning(f"Rate limited by {self.config.name} API. Retrying after {{retry_after}} seconds")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    # Check for successful response
                    response.raise_for_status()
                    
                    # Parse response
                    response_data = response.json()
                    
                    # Track successful call
                    api_metrics()
                    
                    # Track duration
                    duration_ms = (time.time() - start_time) * 1000
                    from backend.core.metrics import external_api_duration_seconds
                    external_api_duration_seconds.labels(
                        provider="{self.config.name}",
                        endpoint=endpoint,
                        status="success",
                        environment="development",
                        version="1.8.0-beta",
                    ).observe(duration_ms / 1000)  # Convert to seconds
                    
                    return response_data
                
            except httpx.HTTPStatusError as e:
                # Check if we should retry
                if e.response.status_code in retry_status_codes and retries < max_retries:
                    retries += 1
                    sleep_time = backoff_factor * (2 ** retries)
                    logger.warning(f"Request failed with status {{e.response.status_code}}. Retrying in {{sleep_time}} seconds...")
                    await asyncio.sleep(sleep_time)
                    continue
                
                # Track failed call
                duration_ms = (time.time() - start_time) * 1000
                from backend.core.metrics import external_api_duration_seconds
                external_api_duration_seconds.labels(
                    provider="{self.config.name}",
                    endpoint=endpoint,
                    status="error",
                    environment="development",
                    version="1.8.0-beta",
                ).observe(duration_ms / 1000)  # Convert to seconds
                
                logger.error(f"Request to {self.config.name} API failed: {{e}}")
                raise
            
            except Exception as e:
                # Track failed call
                duration_ms = (time.time() - start_time) * 1000
                from backend.core.metrics import external_api_duration_seconds
                external_api_duration_seconds.labels(
                    provider="{self.config.name}",
                    endpoint=endpoint,
                    status="error",
                    environment="development",
                    version="1.8.0-beta",
                ).observe(duration_ms / 1000)  # Convert to seconds
                
                logger.error(f"Request to {self.config.name} API failed: {{e}}")
                raise
{"".join(endpoint_methods)}
"""
    
    def _generate_graphql_client(self) -> str:
        """Generate GraphQL client code."""
        class_name = self._to_camel_case(f"{self.config.name}Client")
        
        # Generate query methods
        query_methods = []
        
        # Add default queries if none provided
        if not self.config.endpoints:
            self.config.endpoints = {
                "get_data": """
                    query GetData($id: ID!) {
                        data(id: $id) {
                            id
                            name
                            value
                        }
                    }
                """,
                "get_transactions": """
                    query GetTransactions($address: String!, $limit: Int) {
                        transactions(address: $address, limit: $limit) {
                            hash
                            from
                            to
                            value
                            timestamp
                        }
                    }
                """,
            }
        
        for query_name, query_string in self.config.endpoints.items():
            method_code = f"""
    async def {query_name}(self, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        \"\"\"Execute {query_name} GraphQL query.\"\"\"
        query = \"\"\"
{textwrap.indent(query_string.strip(), ' ' * 12)}
        \"\"\"
        return await self._execute_query(query, variables)
"""
            query_methods.append(method_code)
        
        # Generate client code
        return f"""\"\"\"
{self.config.name.replace('_', ' ').title()} GraphQL Client

This module provides a client for interacting with the {self.config.description or self.config.name} GraphQL API.
It handles authentication, rate limiting, and request retries.

Generated by new_provider_scaffold.py on {datetime.now().strftime('%Y-%m-%d')}
\"\"\"

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import httpx

from backend.core.metrics import ApiMetrics
from backend.providers import get_provider

# Configure module logger
logger = logging.getLogger(__name__)


class {class_name}:
    \"\"\"Client for {self.config.description or self.config.name} GraphQL API.\"\"\"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        \"\"\"
        Initialize the {self.config.name} GraphQL client.
        
        Args:
            api_key: API key for authentication (defaults to environment variable)
            base_url: Base URL for API (defaults to provider registry)
            timeout: Request timeout in seconds
        \"\"\"
        # Get provider configuration
        provider_config = get_provider("{self.config.name}")
        if not provider_config:
            raise ValueError(f"Provider '{self.config.name}' not found in registry")
        
        # Set base URL
        self.base_url = base_url or provider_config.get("base_url", "{self.config.base_url}")
        
        # Set API key
        self.api_key = api_key or os.environ.get(
            "{self.config.name.upper()}_API_KEY",
            provider_config.get("auth", {}).get("key", "")
        )
        
        if not self.api_key and "{self.config.auth_method}" != "none":
            logger.warning(f"No API key provided for {self.config.name}")
        
        # Set timeout
        self.timeout = timeout
        
        # Set up headers
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        
        # Add authentication header if needed
        if self.api_key:
            if "{self.config.auth_method}" == "api_key":
                self.headers["X-API-Key"] = self.api_key
            elif "{self.config.auth_method}" == "bearer_token":
                self.headers["Authorization"] = f"Bearer {{self.api_key}}"
        
        # Add additional headers from provider config
        provider_headers = provider_config.get("headers", {})
        self.headers.update(provider_headers)
        
        # Get rate limit configuration
        self.rate_limit = provider_config.get("rate_limit", {
            "requests_per_minute": {self.config.rate_limit.get('requests_per_minute', 60)},
            "requests_per_day": {self.config.rate_limit.get('requests_per_day', 10000)},
        })
        
        # Get retry configuration
        self.retry_config = provider_config.get("retry", {
            "max_retries": {self.config.retry_config.get('max_retries', 3)},
            "backoff_factor": {self.config.retry_config.get('backoff_factor', 1.0)},
            "retry_status_codes": {self.config.retry_config.get('retry_status_codes', [429, 500, 502, 503, 504])},
        })
        
        logger.info(f"Initialized {self.config.name} GraphQL client with base URL: {{self.base_url}}")
    
    async def _execute_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        \"\"\"
        Execute a GraphQL query.
        
        Args:
            query: GraphQL query string
            variables: Query variables
            
        Returns:
            Response data
            
        Raises:
            httpx.HTTPError: If the request fails
        \"\"\"
        start_time = time.time()
        
        # Prepare request payload
        payload = {
            "query": query,
            "variables": variables or {},
        }
        
        # Track API call
        api_metrics = ApiMetrics.track_call(
            provider="{self.config.name}",
            endpoint="graphql",
            func=lambda: None,
            environment="development",
            version="1.8.0-beta",
        )
        
        # Make request with retries
        retries = 0
        max_retries = self.retry_config.get("max_retries", 3)
        backoff_factor = self.retry_config.get("backoff_factor", 1.0)
        retry_status_codes = self.retry_config.get("retry_status_codes", [429, 500, 502, 503, 504])
        
        while True:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        url=self.base_url,
                        json=payload,
                        headers=self.headers,
                    )
                    
                    # Check for rate limiting
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 1))
                        logger.warning(f"Rate limited by {self.config.name} API. Retrying after {{retry_after}} seconds")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    # Check for successful response
                    response.raise_for_status()
                    
                    # Parse response
                    response_data = response.json()
                    
                    # Check for GraphQL errors
                    if "errors" in response_data:
                        error_message = response_data["errors"][0].get("message", "Unknown GraphQL error")
                        logger.error(f"GraphQL error: {{error_message}}")
                        raise ValueError(f"GraphQL error: {{error_message}}")
                    
                    # Track successful call
                    api_metrics()
                    
                    # Track duration
                    duration_ms = (time.time() - start_time) * 1000
                    from backend.core.metrics import external_api_duration_seconds
                    external_api_duration_seconds.labels(
                        provider="{self.config.name}",
                        endpoint="graphql",
                        status="success",
                        environment="development",
                        version="1.8.0-beta",
                    ).observe(duration_ms / 1000)  # Convert to seconds
                    
                    return response_data.get("data", {})
                
            except httpx.HTTPStatusError as e:
                # Check if we should retry
                if e.response.status_code in retry_status_codes and retries < max_retries:
                    retries += 1
                    sleep_time = backoff_factor * (2 ** retries)
                    logger.warning(f"Request failed with status {{e.response.status_code}}. Retrying in {{sleep_time}} seconds...")
                    await asyncio.sleep(sleep_time)
                    continue
                
                # Track failed call
                duration_ms = (time.time() - start_time) * 1000
                from backend.core.metrics import external_api_duration_seconds
                external_api_duration_seconds.labels(
                    provider="{self.config.name}",
                    endpoint="graphql",
                    status="error",
                    environment="development",
                    version="1.8.0-beta",
                ).observe(duration_ms / 1000)  # Convert to seconds
                
                logger.error(f"Request to {self.config.name} GraphQL API failed: {{e}}")
                raise
            
            except Exception as e:
                # Track failed call
                duration_ms = (time.time() - start_time) * 1000
                from backend.core.metrics import external_api_duration_seconds
                external_api_duration_seconds.labels(
                    provider="{self.config.name}",
                    endpoint="graphql",
                    status="error",
                    environment="development",
                    version="1.8.0-beta",
                ).observe(duration_ms / 1000)  # Convert to seconds
                
                logger.error(f"Request to {self.config.name} GraphQL API failed: {{e}}")
                raise
{"".join(query_methods)}
"""
    
    def _generate_websocket_client(self) -> str:
        """Generate WebSocket client code."""
        class_name = self._to_camel_case(f"{self.config.name}Client")
        
        # Generate subscription methods
        subscription_methods = []
        
        # Add default subscriptions if none provided
        if not self.config.endpoints:
            self.config.endpoints = {
                "subscribe_transactions": {
                    "channel": "transactions",
                    "message": {
                        "type": "subscribe",
                        "channel": "transactions",
                    },
                },
                "subscribe_blocks": {
                    "channel": "blocks",
                    "message": {
                        "type": "subscribe",
                        "channel": "blocks",
                    },
                },
            }
        
        for sub_name, sub_config in self.config.endpoints.items():
            channel = sub_config.get("channel", sub_name)
            message = sub_config.get("message", {"type": "subscribe", "channel": channel})
            
            method_code = f"""
    async def {sub_name}(
        self,
        callback: Callable[[Dict[str, Any]], None],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        \"\"\"
        Subscribe to {channel} channel.
        
        Args:
            callback: Function to call with received messages
            params: Additional parameters for the subscription
        \"\"\"
        message = {message}
        if params:
            message.update(params)
        
        await self._subscribe(message, callback)
"""
            subscription_methods.append(method_code)
        
        # Generate client code
        return f"""\"\"\"
{self.config.name.replace('_', ' ').title()} WebSocket Client

This module provides a client for interacting with the {self.config.description or self.config.name} WebSocket API.
It handles authentication, subscriptions, and message processing.

Generated by new_provider_scaffold.py on {datetime.now().strftime('%Y-%m-%d')}
\"\"\"

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import websockets
from websockets.exceptions import ConnectionClosed

from backend.core.metrics import ApiMetrics
from backend.providers import get_provider

# Configure module logger
logger = logging.getLogger(__name__)


class {class_name}:
    \"\"\"Client for {self.config.description or self.config.name} WebSocket API.\"\"\"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 5,
    ):
        \"\"\"
        Initialize the {self.config.name} WebSocket client.
        
        Args:
            api_key: API key for authentication (defaults to environment variable)
            base_url: WebSocket URL (defaults to provider registry)
            reconnect_interval: Seconds to wait between reconnection attempts
            max_reconnect_attempts: Maximum number of reconnection attempts
        \"\"\"
        # Get provider configuration
        provider_config = get_provider("{self.config.name}")
        if not provider_config:
            raise ValueError(f"Provider '{self.config.name}' not found in registry")
        
        # Set base URL
        self.base_url = base_url or provider_config.get("base_url", "{self.config.base_url}")
        
        # Set API key
        self.api_key = api_key or os.environ.get(
            "{self.config.name.upper()}_API_KEY",
            provider_config.get("auth", {}).get("key", "")
        )
        
        if not self.api_key and "{self.config.auth_method}" != "none":
            logger.warning(f"No API key provided for {self.config.name}")
        
        # Set reconnection parameters
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Set up connection
        self.websocket = None
        self.subscriptions = {}
        self.running = False
        self.reconnect_attempts = 0
        
        logger.info(f"Initialized {self.config.name} WebSocket client with URL: {{self.base_url}}")
    
    async def connect(self) -> None:
        \"\"\"
        Connect to the WebSocket server.
        
        Raises:
            ConnectionError: If connection fails after max attempts
        \"\"\"
        if self.websocket:
            return
        
        self.reconnect_attempts = 0
        self.running = True
        
        while self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                # Connect to WebSocket server
                headers = {}
                if self.api_key:
                    if "{self.config.auth_method}" == "api_key":
                        headers["X-API-Key"] = self.api_key
                    elif "{self.config.auth_method}" == "bearer_token":
                        headers["Authorization"] = f"Bearer {{self.api_key}}"
                
                self.websocket = await websockets.connect(self.base_url, extra_headers=headers)
                logger.info(f"Connected to {self.config.name} WebSocket server")
                
                # Reset reconnect attempts on successful connection
                self.reconnect_attempts = 0
                
                # Start message handler
                asyncio.create_task(self._message_handler())
                
                # Resubscribe to active subscriptions
                for message, callback in self.subscriptions.items():
                    await self._send_message(json.loads(message))
                
                return
            
            except Exception as e:
                self.reconnect_attempts += 1
                logger.error(f"Failed to connect to {self.config.name} WebSocket server: {{e}}")
                
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    self.running = False
                    raise ConnectionError(f"Failed to connect after {{self.max_reconnect_attempts}} attempts")
                
                logger.info(f"Reconnecting in {{self.reconnect_interval}} seconds...")
                await asyncio.sleep(self.reconnect_interval)
    
    async def disconnect(self) -> None:
        \"\"\"Disconnect from the WebSocket server.\"\"\"
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info(f"Disconnected from {self.config.name} WebSocket server")
    
    async def _send_message(self, message: Dict[str, Any]) -> None:
        \"\"\"
        Send a message to the WebSocket server.
        
        Args:
            message: Message to send
            
        Raises:
            ConnectionError: If not connected to the server
        \"\"\"
        if not self.websocket:
            await self.connect()
        
        if not self.websocket:
            raise ConnectionError("Not connected to WebSocket server")
        
        await self.websocket.send(json.dumps(message))
    
    async def _message_handler(self) -> None:
        \"\"\"Handle incoming WebSocket messages.\"\"\"
        if not self.websocket:
            return
        
        try:
            while self.running:
                # Receive message
                message = await self.websocket.recv()
                
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Find matching subscription
                    for sub_message, callback in self.subscriptions.items():
                        sub_data = json.loads(sub_message)
                        channel = sub_data.get("channel")
                        
                        # Check if message matches subscription
                        if channel and data.get("channel") == channel:
                            # Call callback with message data
                            callback(data)
                            
                            # Track message received
                            ApiMetrics.track_call(
                                provider="{self.config.name}",
                                endpoint=f"ws/{{channel}}",
                                func=lambda: None,
                                environment="development",
                                version="1.8.0-beta",
                            )()
                            
                            break
                
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from {self.config.name} WebSocket server")
                
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {{e}}")
        
        except ConnectionClosed:
            logger.warning(f"Connection to {self.config.name} WebSocket server closed")
            
            # Attempt to reconnect
            self.websocket = None
            if self.running:
                asyncio.create_task(self.connect())
        
        except Exception as e:
            logger.error(f"WebSocket message handler error: {{e}}")
            
            # Attempt to reconnect
            self.websocket = None
            if self.running:
                asyncio.create_task(self.connect())
    
    async def _subscribe(
        self,
        message: Dict[str, Any],
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        \"\"\"
        Subscribe to a channel.
        
        Args:
            message: Subscription message
            callback: Function to call with received messages
        \"\"\"
        # Store subscription
        message_key = json.dumps(message, sort_keys=True)
        self.subscriptions[message_key] = callback
        
        # Send subscription message
        await self._send_message(message)
        
        logger.info(f"Subscribed to {{message.get('channel', 'unknown')}} channel")
{"".join(subscription_methods)}
"""
    
    def _generate_custom_client(self) -> str:
        """Generate custom client code."""
        class_name = self._to_camel_case(f"{self.config.name}Client")
        
        # Generate client code
        return f"""\"\"\"
{self.config.name.replace('_', ' ').title()} API Client

This module provides a client for interacting with the {self.config.description or self.config.name} API.
It handles authentication, rate limiting, and request retries.

Generated by new_provider_scaffold.py on {datetime.now().strftime('%Y-%m-%d')}
\"\"\"

import logging
import os
from typing import Any, Dict, List, Optional, Union

from backend.providers import get_provider

# Configure module logger
logger = logging.getLogger(__name__)


class {class_name}:
    \"\"\"Client for {self.config.description or self.config.name} API.\"\"\"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        \"\"\"
        Initialize the {self.config.name} client.
        
        Args:
            api_key: API key for authentication (defaults to environment variable)
            base_url: Base URL for API (defaults to provider registry)
        \"\"\"
        # Get provider configuration
        provider_config = get_provider("{self.config.name}")
        if not provider_config:
            raise ValueError(f"Provider '{self.config.name}' not found in registry")
        
        # Set base URL
        self.base_url = base_url or provider_config.get("base_url", "{self.config.base_url}")
        
        # Set API key
        self.api_key = api_key or os.environ.get(
            "{self.config.name.upper()}_API_KEY",
            provider_config.get("auth", {}).get("key", "")
        )
        
        if not self.api_key and "{self.config.auth_method}" != "none":
            logger.warning(f"No API key provided for {self.config.name}")
        
        logger.info(f"Initialized {self.config.name} client with base URL: {{self.base_url}}")
    
    async def get_data(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        \"\"\"
        Get data from the API.
        
        Args:
            params: Query parameters
            
        Returns:
            Response data
        \"\"\"
        # TODO: Implement API-specific logic
        return {{"message": "Custom client method - implement API-specific logic"}}
"""
    
    def _generate_tool(self, tool_config: Dict[str, Any], tools_dir: Path) -> Path:
        """Generate a tool class."""
        tool_name = tool_config["name"]
        tool_description = tool_config.get("description", f"Tool for {self.config.name}")
        endpoints = tool_config.get("endpoints", ["get_data"])
        data_types = tool_config.get("data_types", self.config.data_types)
        
        # Create tool file path
        tool_file = tools_dir / f"{tool_name}.py"
        
        # Check if file already exists
        if tool_file.exists() and not self.force:
            raise ValueError(f"Tool file already exists: {tool_file}. Use --force to overwrite.")
        
        # Generate tool class name
        class_name = self._to_camel_case(tool_name)
        if not class_name.endswith("Tool"):
            class_name += "Tool"
        
        # Generate method implementations
        method_impls = []
        for endpoint in endpoints:
            method_impl = f"""
    async def {endpoint}(self, params: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        Call {endpoint} endpoint.
        
        Args:
            params: Request parameters
            
        Returns:
            Response data
        \"\"\"
        return await self.client.{endpoint}(params)
"""
            method_impls.append(method_impl)
        
        # Generate execute method
        execute_method = """
    async def _execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given parameters."""
        # Validate parameters
        self._validate_parameters(params)
        
        # Determine which endpoint to call based on parameters
        if "method" in params:
            method = params.pop("method")
            
            # Call the appropriate method
            if hasattr(self, method) and callable(getattr(self, method)):
                return await getattr(self, method)(params)
            else:
                raise ValueError(f"Unknown method: {method}")
"""
        
        # Add default method call if there's only one endpoint
        if len(endpoints) == 1:
            execute_method += f"""
        # Default to {endpoints[0]} if no method specified
        return await self.{endpoints[0]}(params)
"""
        else:
            execute_method += """
        # No method specified, raise error
        raise ValueError("Method parameter is required")
"""
        
        # Generate tool code
        tool_code = f"""\"\"\"
{tool_name.replace('_', ' ').title()} Tool

This module provides a tool for interacting with the {self.config.description or self.config.name} API.

Generated by new_provider_scaffold.py on {datetime.now().strftime('%Y-%m-%d')}
\"\"\"

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from backend.agents.tools.base_tool import AbstractApiTool
from backend.integrations.{self.config.name}_client import {self._to_camel_case(f"{self.config.name}Client")}

# Configure module logger
logger = logging.getLogger(__name__)


class {class_name}Request(BaseModel):
    \"\"\"Request model for {class_name}.\"\"\"
    # Common parameters
    method: Optional[str] = Field(
        None,
        description="Method to call (e.g., {', '.join(endpoints)})",
    )
    
    # Method-specific parameters can be added here
    # For example:
    # address: Optional[str] = Field(None, description="Blockchain address")
    # chain: Optional[str] = Field(None, description="Blockchain network")
    
    @validator("method")
    def validate_method(cls, v):
        \"\"\"Validate method parameter.\"\"\"
        if v is not None and v not in {endpoints}:
            raise ValueError(f"Invalid method: {{v}}. Must be one of: {', '.join(endpoints)}")
        return v


class {class_name}(AbstractApiTool):
    \"\"\"Tool for interacting with {self.config.description or self.config.name} API.\"\"\"
    
    name = "{tool_name}"
    description = "{tool_description}"
    provider_id = "{self.config.name}"
    data_types = {data_types}
    request_model = {class_name}Request
    
    def __init__(self):
        \"\"\"Initialize the tool.\"\"\"
        super().__init__()
        self.client = {self._to_camel_case(f"{self.config.name}Client")}()
{execute_method}{"".join(method_impls)}
"""
        
        # Write tool code to file
        with open(tool_file, 'w') as f:
            f.write(tool_code)
        
        print(f"✅ Generated tool: {tool_file}")
        return tool_file
    
    def _generate_client_test(self) -> str:
        """Generate test code for client."""
        client_class = self._to_camel_case(f"{self.config.name}Client")
        
        # Generate test code
        return f"""\"\"\"
Tests for {self.config.name}_client.py

Generated by new_provider_scaffold.py on {datetime.now().strftime('%Y-%m-%d')}
\"\"\"

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.integrations.{self.config.name}_client import {client_class}


@pytest.fixture
def mock_provider_config():
    \"\"\"Mock provider configuration.\"\"\"
    return {{
        "id": "{self.config.name}",
        "type": "{self.config.provider_type}",
        "base_url": "{self.config.base_url}",
        "auth": {{
            "method": "{self.config.auth_method}",
            "key": "test_api_key",
        }},
        "rate_limit": {{
            "requests_per_minute": 60,
            "requests_per_day": 10000,
        }},
        "retry": {{
            "max_retries": 3,
            "backoff_factor": 1.0,
            "retry_status_codes": [429, 500, 502, 503, 504],
        }},
    }}


@pytest.fixture
def client(mock_provider_config):
    \"\"\"Create a client instance with mocked provider config.\"\"\"
    with patch("backend.integrations.{self.config.name}_client.get_provider", return_value=mock_provider_config):
        client = {client_class}(api_key="test_api_key")
        return client


@pytest.mark.asyncio
async def test_client_initialization(client):
    \"\"\"Test client initialization.\"\"\"
    assert client.base_url == "{self.config.base_url}"
    assert client.api_key == "test_api_key"


@pytest.mark.asyncio
async def test_client_without_api_key():
    \"\"\"Test client initialization without API key.\"\"\"
    with patch("backend.integrations.{self.config.name}_client.get_provider", return_value={{
        "id": "{self.config.name}",
        "type": "{self.config.provider_type}",
        "base_url": "{self.config.base_url}",
    }}):
        with patch.dict(os.environ, {{"{self.config.name.upper()}_API_KEY": "env_api_key"}}):
            client = {client_class}()
            assert client.api_key == "env_api_key"
"""
    
    def _generate_tool_test(self, tool_config: Dict[str, Any]) -> str:
        """Generate test code for tool."""
        tool_name = tool_config["name"]
        tool_class = self._to_camel_case(tool_name)
        if not tool_class.endswith("Tool"):
            tool_class += "Tool"
        
        client_class = self._to_camel_case(f"{self.config.name}Client")
        
        # Generate test code
        return f"""\"\"\"
Tests for {tool_name}.py

Generated by new_provider_scaffold.py on {datetime.now().strftime('%Y-%m-%d')}
\"\"\"

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.tools.{self.config.name}.{tool_name} import {tool_class}


@pytest.fixture
def mock_client():
    \"\"\"Mock {self.config.name} client.\"\"\"
    client = AsyncMock()
    client.get_data = AsyncMock(return_value={{"result": "test_data"}})
    return client


@pytest.fixture
def tool(mock_client):
    \"\"\"Create a tool instance with mocked client.\"\"\"
    with patch(f"backend.agents.tools.{self.config.name}.{tool_name}.{client_class}", return_value=mock_client):
        tool = {tool_class}()
        return tool


@pytest.mark.asyncio
async def test_tool_initialization(tool):
    \"\"\"Test tool initialization.\"\"\"
    assert tool.name == "{tool_name}"
    assert tool.provider_id == "{self.config.name}"


@pytest.mark.asyncio
async def test_tool_execution(tool, mock_client):
    \"\"\"Test tool execution.\"\"\"
    params = {{"param1": "value1"}}
    result = await tool._execute(params)
    
    assert result == {{"result": "test_data"}}
    mock_client.get_data.assert_called_once_with(params)


@pytest.mark.asyncio
async def test_tool_validation(tool):
    \"\"\"Test tool parameter validation.\"\"\"
    # This test depends on the specific validation rules of your tool
    # Modify as needed based on your tool's requirements
    params = {{"method": "invalid_method"}}
    
    with pytest.raises(ValueError):
        await tool._execute(params)
"""
    
    def _generate_integration_test(self) -> str:
        """Generate integration test code."""
        return f"""\"\"\"
Integration tests for {self.config.name} provider.

Generated by new_provider_scaffold.py on {datetime.now().strftime('%Y-%m-%d')}
\"\"\"

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.integrations.{self.config.name}_client import {self._to_camel_case(f"{self.config.name}Client")}
from backend.core.neo4j_loader import Neo4jLoader


# Skip these tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.environ.get("{self.config.name.upper()}_API_KEY"),
    reason="{self.config.name} API key not available",
)


@pytest.fixture
def mock_neo4j():
    \"\"\"Mock Neo4j loader.\"\"\"
    loader = AsyncMock(spec=Neo4jLoader)
    loader._execute_query = AsyncMock()
    loader._process_result_stats = MagicMock(return_value=MagicMock(
        nodes_created=0,
        relationships_created=0,
        properties_set=0,
        labels_added=0,
    ))
    return loader


@pytest.mark.integration
@pytest.mark.asyncio
async def test_client_real_connection():
    \"\"\"Test connecting to the real API (requires API key).\"\"\"
    # This test will be skipped if no API key is available
    client = {self._to_camel_case(f"{self.config.name}Client")}()
    
    # Test a simple API call - modify based on your API
    try:
        result = await client.get_data({{
            # Add required parameters here
        }})
        assert result is not None
    except Exception as e:
        pytest.fail(f"API call failed: {{e}}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_neo4j_integration(mock_neo4j):
    \"\"\"Test integration with Neo4j.\"\"\"
    client = {self._to_camel_case(f"{self.config.name}Client")}()
    
    # Mock API response
    with patch.object(client, 'get_data', return_value={{
        "data": [
            {{"id": "1", "name": "Test", "value": 100}}
        ]
    }}):
        # Get data from API
        data = await client.get_data({{}})
        
        # Process data with Neo4j
        # This is a simplified example - modify based on your actual data model
        for item in data.get("data", []):
            query = \"\"\"
            MERGE (n:Node {{id: $id}})
            SET n.name = $name, n.value = $value
            RETURN n
            \"\"\"
            params = {{
                "id": item["id"],
                "name": item["name"],
                "value": item["value"],
            }}
            
            await mock_neo4j._execute_query(query, params)
        
        # Verify Neo4j was called
        assert mock_neo4j._execute_query.called


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_error_handling():
    \"\"\"Test error handling for API calls.\"\"\"
    client = {self._to_camel_case(f"{self.config.name}Client")}()
    
    # Test with invalid parameters
    with pytest.raises(Exception):
        await client.get_data({{
            "invalid_param": "this_should_cause_an_error"
        }})
"""
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate scaffolding for new API providers and tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
            python scripts/new_provider_scaffold.py
            python scripts/new_provider_scaffold.py --name "example_provider" --type "rest"
            python scripts/new_provider_scaffold.py --config provider_config.json
        """),
    )
    
    parser.add_argument("--name", help="Provider name (snake_case)")
    parser.add_argument("--type", choices=PROVIDER_TYPES, help="Provider type")
    parser.add_argument("--config", help="Path to JSON configuration file")
    parser.add_argument("--output-dir", help="Output directory for generated files")
    parser.add_argument("--skip-tests", action="store_true", help="Skip generating test files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    
    return parser.parse_args()


def prompt_provider_config() -> ProviderConfig:
    """Prompt user for provider configuration."""
    print("📋 Provider Configuration Wizard")
    print("===============================")
    
    # Provider name
    while True:
        name = input("Provider name (snake_case): ").strip()
        if name and re.match(r'^[a-z][a-z0-9_]*$', name):
            break
        print("❌ Invalid name. Use snake_case (e.g., 'example_provider').")
    
    # Provider type
    print("\nProvider types:")
    for i, provider_type in enumerate(PROVIDER_TYPES):
        print(f"  {i+1}. {provider_type}")
    
    while True:
        type_input = input(f"Provider type (1-{len(PROVIDER_TYPES)}): ").strip()
        try:
            type_index = int(type_input) - 1
            if 0 <= type_index < len(PROVIDER_TYPES):
                provider_type = PROVIDER_TYPES[type_index]
                break
        except ValueError:
            pass
        print(f"❌ Invalid choice. Enter a number between 1 and {len(PROVIDER_TYPES)}.")
    
    # Base URL
    while True:
        base_url = input("Base URL: ").strip()
        if base_url:
            break
        print("❌ Base URL is required.")
    
    # Authentication method
    print("\nAuthentication methods:")
    for i, auth_method in enumerate(AUTH_METHODS):
        print(f"  {i+1}. {auth_method}")
    
    while True:
        auth_input = input(f"Authentication method (1-{len(AUTH_METHODS)}): ").strip()
        try:
            auth_index = int(auth_input) - 1
            if 0 <= auth_index < len(AUTH_METHODS):
                auth_method = AUTH_METHODS[auth_index]
                break
        except ValueError:
            pass
        print(f"❌ Invalid choice. Enter a number between 1 and {len(AUTH_METHODS)}.")
    
    # Data types
    print("\nData types (comma-separated list):")
    for i, data_type in enumerate(DATA_TYPES):
        print(f"  {i+1}. {data_type}")
    
    while True:
        data_types_input = input("Data types (e.g., '1,3,5' or '1-3,5'): ").strip()
        if not data_types_input:
            data_types = []
            break
        
        try:
            # Parse input like "1,3,5" or "1-3,5"
            data_type_indices = set()
            for part in data_types_input.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    data_type_indices.update(range(start - 1, end))
                else:
                    data_type_indices.add(int(part) - 1)
            
            # Validate indices
            if all(0 <= i < len(DATA_TYPES) for i in data_type_indices):
                data_types = [DATA_TYPES[i] for i in data_type_indices]
                break
        except ValueError:
            pass
        
        print(f"❌ Invalid format. Use comma-separated numbers or ranges (e.g., '1,3,5' or '1-3,5').")
    
    # Description
    description = input("\nDescription (optional): ").strip()
    
    # Create provider config
    return ProviderConfig(
        name=name,
        provider_type=provider_type,
        base_url=base_url,
        auth_method=auth_method,
        data_types=data_types,
        description=description,
    )


def main():
    """Main function."""
    args = parse_args()
    
    # Get provider configuration
    if args.config:
        # Load from config file
        try:
            config = ProviderConfig.from_json_file(args.config)
            print(f"📋 Loaded configuration from {args.config}")
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return 1
    elif args.name and args.type:
        # Create from command line arguments
        config = ProviderConfig(
            name=args.name,
            provider_type=args.type,
            base_url=input("Base URL: ").strip(),
            auth_method="api_key",  # Default
            data_types=[],  # Empty by default
        )
    else:
        # Prompt for configuration
        config = prompt_provider_config()
    
    # Create output paths
    paths = DEFAULT_PATHS.copy()
    if args.output_dir:
        for key in paths:
            paths[key] = Path(args.output_dir) / paths[key]
    
    # Generate scaffolding
    try:
        generator = ScaffoldGenerator(
            config=config,
            paths=paths,
            force=args.force,
            skip_tests=args.skip_tests,
        )
        
        generated_files = generator.generate_all()
        
        print("\n✅ Successfully generated scaffolding!")
        print(f"Provider: {config.name} ({config.provider_type})")
        
        # Save configuration for future reference
        config_path = Path(f"{config.name}_config.json")
        config.to_json_file(config_path)
        print(f"📋 Saved configuration to {config_path}")
        
        # Print next steps
        print("\n📝 Next Steps:")
        print(f"1. Update the client in {generated_files['client_integration']}")
        print(f"2. Implement the tool methods in {generated_files['tools'][1:]}")
        if not args.skip_tests:
            print(f"3. Update tests in {generated_files['tests']}")
        print(f"4. Add environment variable {config.name.upper()}_API_KEY to .env file")
        
        return 0
    
    except Exception as e:
        print(f"❌ Error generating scaffolding: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
