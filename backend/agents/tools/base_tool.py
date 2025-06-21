"""
Abstract API Tool Base Class

This module provides a comprehensive base class for all API-based tools in the system,
with built-in support for retries, metrics, caching, rate limiting, and error handling.
All SIM tools and other API-based tools should inherit from this class.
"""

import asyncio
import functools
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

import aiohttp
import backoff
import httpx
import redis
from pydantic import BaseModel, Field
from tenacity import (
    RetryError,
    Retrying,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
)

from backend.core.events import publish_event
from backend.core.metrics import ApiMetrics
from backend.providers import ProviderConfig, get_provider

# Type variables for generic typing
T = TypeVar("T")
ResponseType = TypeVar("ResponseType")
RequestType = TypeVar("RequestType", bound=BaseModel)

# Configure module logger
logger = logging.getLogger(__name__)


class ApiRequestMethod(str, Enum):
    """HTTP methods supported by the API tool."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class ApiError(Exception):
    """Base exception for API-related errors."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        provider_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        response_body: Optional[str] = None
    ):
        self.message = message
        self.status_code = status_code
        self.provider_id = provider_id
        self.endpoint = endpoint
        self.response_body = response_body
        super().__init__(message)


class RateLimitError(ApiError):
    """Exception raised when rate limits are exceeded."""
    pass


class AuthenticationError(ApiError):
    """Exception raised for authentication failures."""
    pass


class ConnectionError(ApiError):
    """Exception raised for connection failures."""
    pass


class TimeoutError(ApiError):
    """Exception raised for request timeouts."""
    pass


class ApiResponse(Generic[ResponseType]):
    """
    Generic API response wrapper with metadata.
    
    Attributes:
        data: The parsed response data
        status_code: HTTP status code
        headers: Response headers
        elapsed: Request duration in seconds
        cached: Whether the response was served from cache
        retries: Number of retries performed
        provider_id: ID of the provider that served the request
        endpoint: The endpoint that was called
        timestamp: When the request was made
    """
    
    def __init__(
        self,
        data: ResponseType,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        elapsed: float = 0.0,
        cached: bool = False,
        retries: int = 0,
        provider_id: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        self.data = data
        self.status_code = status_code
        self.headers = headers or {}
        self.elapsed = elapsed
        self.cached = cached
        self.retries = retries
        self.provider_id = provider_id
        self.endpoint = endpoint
        self.timestamp = datetime.now()
    
    def __repr__(self) -> str:
        return (
            f"ApiResponse(status_code={self.status_code}, "
            f"elapsed={self.elapsed:.3f}s, cached={self.cached}, "
            f"provider={self.provider_id}, endpoint={self.endpoint})"
        )


class ApiRequest(BaseModel):
    """Base model for API requests with common fields."""
    
    provider_id: Optional[str] = Field(
        None, description="Provider ID to use for this request"
    )
    cache_ttl: Optional[int] = Field(
        None, description="Cache TTL in seconds, None for no caching"
    )
    timeout: Optional[float] = Field(
        None, description="Request timeout in seconds"
    )
    max_retries: Optional[int] = Field(
        None, description="Maximum number of retries"
    )


class AbstractApiTool(ABC, Generic[RequestType, ResponseType]):
    """
    Abstract base class for all API-based tools.
    
    This class provides:
    - Common retry/back-off logic with exponential backoff
    - Prometheus metrics integration for timing and cost tracking
    - Provider registry integration
    - Rate limiting support
    - Authentication handling
    - Error handling with proper logging
    - Request/response caching capabilities
    
    All SIM tools and other API-based tools should inherit from this class.
    """
    
    # Class configuration
    name: str = "abstract_api_tool"
    description: str = "Abstract base class for API tools"
    provider_id: str = "abstract"
    default_cache_ttl: int = 300  # 5 minutes
    default_timeout: float = 30.0  # 30 seconds
    default_max_retries: int = 3
    request_model: Type[RequestType] = cast(Type[RequestType], ApiRequest)
    response_model: Type[ResponseType] = cast(Type[ResponseType], Dict[str, Any])
    
    def __init__(
        self,
        provider_id: Optional[str] = None,
        cache_client: Optional[redis.Redis] = None,
        environment: str = "development",
        version: str = "1.8.0-beta",
    ):
        """
        Initialize the API tool.
        
        Args:
            provider_id: Override the default provider ID
            cache_client: Redis client for caching (optional)
            environment: Environment name for metrics
            version: Application version for metrics
        """
        self.provider_id = provider_id or self.provider_id
        self.cache_client = cache_client
        self.environment = environment
        self.version = version
        self.provider_config = self._load_provider_config()
        
        # Initialize HTTP client
        timeout = httpx.Timeout(
            connect=self.provider_config.get("timeout", {}).get("connect_seconds", 5.0),
            read=self.provider_config.get("timeout", {}).get("read_seconds", 30.0),
            total=self.provider_config.get("timeout", {}).get("total_seconds", 60.0),
        )
        self.client = httpx.Client(timeout=timeout)
        
        # Initialize async HTTP client
        self.async_client = httpx.AsyncClient(timeout=timeout)
        
        logger.debug(f"Initialized {self.name} with provider {self.provider_id}")
    
    def _load_provider_config(self) -> Dict[str, Any]:
        """
        Load provider configuration from the registry.
        
        Returns:
            Provider configuration dictionary
            
        Raises:
            ValueError: If the provider is not found
        """
        provider = get_provider(self.provider_id)
        if not provider:
            logger.error(f"Provider not found: {self.provider_id}")
            raise ValueError(f"Provider not found: {self.provider_id}")
        
        logger.debug(f"Loaded configuration for provider: {self.provider_id}")
        return cast(Dict[str, Any], provider)
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers based on provider configuration.
        
        Returns:
            Dictionary of authentication headers
            
        Raises:
            AuthenticationError: If required auth environment variables are missing
        """
        auth_config = self.provider_config.get("auth", {})
        if not auth_config:
            return {}
        
        auth_type = auth_config.get("type")
        headers = {}
        
        if auth_type == "api_key":
            header_name = auth_config.get("header_name")
            key_env_var = auth_config.get("key_env_var")
            key_prefix = auth_config.get("key_prefix", "")
            
            if header_name and key_env_var:
                import os
                api_key = os.environ.get(key_env_var)
                
                if not api_key and not auth_config.get("optional", False):
                    raise AuthenticationError(
                        f"Missing required API key environment variable: {key_env_var}",
                        provider_id=self.provider_id
                    )
                
                if api_key:
                    headers[header_name] = f"{key_prefix}{api_key}"
        
        elif auth_type == "basic":
            # Basic auth is handled differently in the request itself
            pass
        
        return headers
    
    def _get_auth_params(self) -> Dict[str, str]:
        """
        Get authentication query parameters based on provider configuration.
        
        Returns:
            Dictionary of authentication query parameters
        """
        auth_config = self.provider_config.get("auth", {})
        if not auth_config:
            return {}
        
        auth_type = auth_config.get("type")
        params = {}
        
        if auth_type == "api_key" and auth_config.get("query_param"):
            query_param = auth_config.get("query_param")
            key_env_var = auth_config.get("key_env_var")
            
            if query_param and key_env_var:
                import os
                api_key = os.environ.get(key_env_var)
                
                if not api_key and not auth_config.get("optional", False):
                    raise AuthenticationError(
                        f"Missing required API key environment variable: {key_env_var}",
                        provider_id=self.provider_id
                    )
                
                if api_key:
                    params[query_param] = api_key
        
        return params
    
    def _get_base_url(self) -> str:
        """
        Get the base URL for the provider.
        
        Returns:
            Base URL string
            
        Raises:
            ValueError: If the base URL is not configured
        """
        base_url = self.provider_config.get("base_url")
        if not base_url:
            raise ValueError(f"Base URL not configured for provider: {self.provider_id}")
        
        return base_url
    
    def _get_endpoint_url(self, endpoint: str) -> str:
        """
        Get the full URL for an endpoint.
        
        Args:
            endpoint: The endpoint path
            
        Returns:
            Full URL string
        """
        base_url = self._get_base_url()
        
        # Handle both absolute and relative endpoints
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        
        # Ensure base_url doesn't end with slash and endpoint starts with slash
        base_url = base_url.rstrip("/")
        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        
        return f"{base_url}{endpoint}"
    
    def _get_cache_key(self, endpoint: str, params: Dict[str, Any], data: Any) -> str:
        """
        Generate a cache key for the request.
        
        Args:
            endpoint: The endpoint path
            params: Query parameters
            data: Request body data
            
        Returns:
            Cache key string
        """
        # Create a dictionary with all request components
        key_parts = {
            "provider_id": self.provider_id,
            "endpoint": endpoint,
            "params": params,
            "data": data,
        }
        
        # Convert to a stable string representation
        key_str = json.dumps(key_parts, sort_keys=True)
        
        # Use a hash to keep the key length reasonable
        import hashlib
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"api_cache:{self.name}:{key_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[ApiResponse[ResponseType]]:
        """
        Try to get a response from cache.
        
        Args:
            cache_key: The cache key
            
        Returns:
            Cached API response or None if not found
        """
        if not self.cache_client:
            return None
        
        try:
            cached_data = self.cache_client.get(cache_key)
            if cached_data:
                response = json.loads(cached_data)
                logger.debug(f"Cache hit for {cache_key}")
                
                # Reconstruct ApiResponse object
                return ApiResponse(
                    data=cast(ResponseType, response.get("data")),
                    status_code=response.get("status_code", 200),
                    headers=response.get("headers", {}),
                    elapsed=response.get("elapsed", 0.0),
                    cached=True,
                    retries=response.get("retries", 0),
                    provider_id=response.get("provider_id"),
                    endpoint=response.get("endpoint"),
                )
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, response: ApiResponse[ResponseType], ttl: int) -> None:
        """
        Save a response to cache.
        
        Args:
            cache_key: The cache key
            response: The API response to cache
            ttl: Cache TTL in seconds
        """
        if not self.cache_client:
            return
        
        try:
            # Convert ApiResponse to a serializable dictionary
            cache_data = {
                "data": response.data,
                "status_code": response.status_code,
                "headers": response.headers,
                "elapsed": response.elapsed,
                "retries": response.retries,
                "provider_id": response.provider_id,
                "endpoint": response.endpoint,
                "timestamp": response.timestamp.isoformat(),
            }
            
            self.cache_client.setex(
                cache_key,
                ttl,
                json.dumps(cache_data, default=str)
            )
            logger.debug(f"Saved to cache: {cache_key} (TTL: {ttl}s)")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def _should_retry(self, exception: Exception) -> bool:
        """
        Determine if a request should be retried based on the exception.
        
        Args:
            exception: The exception that was raised
            
        Returns:
            True if the request should be retried, False otherwise
        """
        # Always retry on connection errors
        if isinstance(exception, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        
        # Retry on rate limit errors with appropriate backoff
        if isinstance(exception, RateLimitError):
            return True
        
        # Retry on server errors (5xx)
        if isinstance(exception, ApiError) and exception.status_code and exception.status_code >= 500:
            return True
        
        # Check provider-specific retry configuration
        retry_config = self.provider_config.get("retry", {})
        retry_status_codes = retry_config.get("retry_on_status_codes", [429, 500, 502, 503, 504])
        
        if isinstance(exception, ApiError) and exception.status_code in retry_status_codes:
            return True
        
        return False
    
    def _get_retry_config(self, max_retries: Optional[int] = None) -> Dict[str, Any]:
        """
        Get retry configuration based on provider settings.
        
        Args:
            max_retries: Override the default max retries
            
        Returns:
            Dictionary with retry configuration
        """
        retry_config = self.provider_config.get("retry", {})
        
        return {
            "max_attempts": max_retries or retry_config.get("max_attempts", self.default_max_retries),
            "initial_backoff": retry_config.get("initial_backoff_seconds", 1.0),
            "max_backoff": retry_config.get("max_backoff_seconds", 30.0),
            "backoff_factor": retry_config.get("backoff_factor", 2.0),
        }
    
    def _track_request_metrics(
        self,
        endpoint: str,
        start_time: float,
        status: str = "success",
        credit_type: Optional[str] = None,
        credit_amount: Optional[float] = None,
    ) -> None:
        """
        Track metrics for an API request.
        
        Args:
            endpoint: The endpoint that was called
            start_time: Request start time (from time.time())
            status: Request status (success/error)
            credit_type: Type of credits used (optional)
            credit_amount: Amount of credits used (optional)
        """
        duration = time.time() - start_time
        
        # Track API call metrics
        ApiMetrics.track_call(
            provider=self.provider_id,
            endpoint=endpoint,
            func=lambda: None,
            environment=self.environment,
            version=self.version,
        )()
        
        # Track API duration
        from backend.core.metrics import external_api_duration_seconds
        external_api_duration_seconds.labels(
            provider=self.provider_id,
            endpoint=endpoint,
            status=status,
            environment=self.environment,
            version=self.version,
        ).observe(duration)
        
        # Track credit usage if provided
        if credit_type and credit_amount:
            ApiMetrics.track_credits(
                provider=self.provider_id,
                endpoint=endpoint,
                credit_type=credit_type,
                amount=credit_amount,
                environment=self.environment,
                version=self.version,
                status=status,
            )
        
        # Publish event for other components to react to
        publish_event("api_request", {
            "provider_id": self.provider_id,
            "endpoint": endpoint,
            "duration": duration,
            "status": status,
            "credit_type": credit_type,
            "credit_amount": credit_amount,
        })
    
    def _handle_response(
        self,
        response: httpx.Response,
        endpoint: str,
    ) -> ResponseType:
        """
        Handle an HTTP response, raising appropriate exceptions for errors.
        
        Args:
            response: The HTTP response
            endpoint: The endpoint that was called
            
        Returns:
            Parsed response data
            
        Raises:
            ApiError: For API errors
            RateLimitError: For rate limit errors
            AuthenticationError: For authentication errors
        """
        # Handle different status codes
        if response.status_code == 200:
            try:
                return cast(ResponseType, response.json())
            except ValueError:
                # If not JSON, return text or bytes
                return cast(ResponseType, response.text)
        
        # Handle error responses
        error_body = response.text
        
        if response.status_code == 429:
            raise RateLimitError(
                f"Rate limit exceeded for {self.provider_id}",
                status_code=response.status_code,
                provider_id=self.provider_id,
                endpoint=endpoint,
                response_body=error_body,
            )
        
        if response.status_code in (401, 403):
            raise AuthenticationError(
                f"Authentication failed for {self.provider_id}",
                status_code=response.status_code,
                provider_id=self.provider_id,
                endpoint=endpoint,
                response_body=error_body,
            )
        
        # Generic error for other status codes
        raise ApiError(
            f"API error: {response.status_code} - {error_body[:100]}",
            status_code=response.status_code,
            provider_id=self.provider_id,
            endpoint=endpoint,
            response_body=error_body,
        )
    
    def request(
        self,
        endpoint: str,
        method: ApiRequestMethod = ApiRequestMethod.GET,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_ttl: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> ApiResponse[ResponseType]:
        """
        Make an API request with retries, caching, and metrics.
        
        Args:
            endpoint: The endpoint to call
            method: HTTP method to use
            params: Query parameters
            data: Request body data
            headers: Additional headers
            cache_ttl: Cache TTL in seconds (None for default, 0 for no caching)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            
        Returns:
            API response wrapper
            
        Raises:
            ApiError: For API errors
            ConnectionError: For connection errors
            TimeoutError: For timeouts
        """
        params = params or {}
        headers = headers or {}
        
        # Determine if we should use caching
        use_cache = self.cache_client is not None and cache_ttl != 0
        effective_ttl = cache_ttl or self.default_cache_ttl
        
        # Generate cache key if using cache
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(endpoint, params, data)
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Prepare request
        url = self._get_endpoint_url(endpoint)
        all_headers = {**self._get_auth_headers(), **headers}
        all_params = {**self._get_auth_params(), **params}
        
        # Set up retry configuration
        retry_config = self._get_retry_config(max_retries)
        retries = 0
        
        # Track metrics
        start_time = time.time()
        
        try:
            # Use tenacity for retries with exponential backoff
            for attempt in Retrying(
                stop=(
                    stop_after_attempt(retry_config["max_attempts"]) | 
                    stop_after_delay(timeout or self.default_timeout)
                ),
                wait=wait_exponential(
                    multiplier=retry_config["initial_backoff"],
                    max=retry_config["max_backoff"],
                    exp_base=retry_config["backoff_factor"],
                ),
                retry=(
                    retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout)) | 
                    retry_if_exception_type(RateLimitError) |
                    retry_if_exception_type(ApiError)
                ),
                reraise=True,
            ):
                with attempt:
                    retries = attempt.retry_state.attempt_number - 1
                    
                    # Make the request
                    with self.client.stream(
                        method=method.value,
                        url=url,
                        params=all_params,
                        json=data if method != ApiRequestMethod.GET else None,
                        headers=all_headers,
                        timeout=timeout or self.default_timeout,
                    ) as response:
                        # Process the response
                        response_data = self._handle_response(response, endpoint)
                        
                        # Create API response wrapper
                        api_response = ApiResponse(
                            data=response_data,
                            status_code=response.status_code,
                            headers=dict(response.headers),
                            elapsed=time.time() - start_time,
                            cached=False,
                            retries=retries,
                            provider_id=self.provider_id,
                            endpoint=endpoint,
                        )
                        
                        # Cache the response if needed
                        if use_cache and cache_key:
                            self._save_to_cache(cache_key, api_response, effective_ttl)
                        
                        # Track metrics
                        self._track_request_metrics(
                            endpoint=endpoint,
                            start_time=start_time,
                            status="success",
                        )
                        
                        return api_response
        
        except RetryError as e:
            # Handle the case when all retries are exhausted
            logger.error(f"All retries failed for {self.provider_id} - {endpoint}: {e}")
            self._track_request_metrics(endpoint=endpoint, start_time=start_time, status="error")
            
            # Re-raise the original exception
            if e.__cause__:
                raise e.__cause__
            
            raise ApiError(
                f"Request failed after {retries} retries: {e}",
                provider_id=self.provider_id,
                endpoint=endpoint,
            )
        
        except httpx.ConnectError as e:
            logger.error(f"Connection error for {self.provider_id} - {endpoint}: {e}")
            self._track_request_metrics(endpoint=endpoint, start_time=start_time, status="error")
            
            raise ConnectionError(
                f"Failed to connect to {self.provider_id}: {e}",
                provider_id=self.provider_id,
                endpoint=endpoint,
            )
        
        except httpx.TimeoutException as e:
            logger.error(f"Timeout for {self.provider_id} - {endpoint}: {e}")
            self._track_request_metrics(endpoint=endpoint, start_time=start_time, status="error")
            
            raise TimeoutError(
                f"Request to {self.provider_id} timed out: {e}",
                provider_id=self.provider_id,
                endpoint=endpoint,
            )
        
        except Exception as e:
            logger.error(f"Unexpected error for {self.provider_id} - {endpoint}: {e}")
            self._track_request_metrics(endpoint=endpoint, start_time=start_time, status="error")
            
            # Re-raise as ApiError
            raise ApiError(
                f"API request failed: {e}",
                provider_id=self.provider_id,
                endpoint=endpoint,
            )
    
    async def async_request(
        self,
        endpoint: str,
        method: ApiRequestMethod = ApiRequestMethod.GET,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_ttl: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> ApiResponse[ResponseType]:
        """
        Make an async API request with retries, caching, and metrics.
        
        Args:
            endpoint: The endpoint to call
            method: HTTP method to use
            params: Query parameters
            data: Request body data
            headers: Additional headers
            cache_ttl: Cache TTL in seconds (None for default, 0 for no caching)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            
        Returns:
            API response wrapper
            
        Raises:
            ApiError: For API errors
            ConnectionError: For connection errors
            TimeoutError: For timeouts
        """
        params = params or {}
        headers = headers or {}
        
        # Determine if we should use caching
        use_cache = self.cache_client is not None and cache_ttl != 0
        effective_ttl = cache_ttl or self.default_cache_ttl
        
        # Generate cache key if using cache
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(endpoint, params, data)
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Prepare request
        url = self._get_endpoint_url(endpoint)
        all_headers = {**self._get_auth_headers(), **headers}
        all_params = {**self._get_auth_params(), **params}
        
        # Set up retry configuration
        retry_config = self._get_retry_config(max_retries)
        retries = 0
        max_attempts = retry_config["max_attempts"]
        
        # Track metrics
        start_time = time.time()
        
        # Manual retry loop for async
        for attempt in range(max_attempts):
            try:
                retries = attempt
                
                # Make the request
                async with self.async_client.stream(
                    method=method.value,
                    url=url,
                    params=all_params,
                    json=data if method != ApiRequestMethod.GET else None,
                    headers=all_headers,
                    timeout=timeout or self.default_timeout,
                ) as response:
                    # Process the response
                    response_data = self._handle_response(response, endpoint)
                    
                    # Create API response wrapper
                    api_response = ApiResponse(
                        data=response_data,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        elapsed=time.time() - start_time,
                        cached=False,
                        retries=retries,
                        provider_id=self.provider_id,
                        endpoint=endpoint,
                    )
                    
                    # Cache the response if needed
                    if use_cache and cache_key:
                        self._save_to_cache(cache_key, api_response, effective_ttl)
                    
                    # Track metrics
                    self._track_request_metrics(
                        endpoint=endpoint,
                        start_time=start_time,
                        status="success",
                    )
                    
                    return api_response
            
            except Exception as e:
                # Check if we should retry
                if not self._should_retry(e) or attempt == max_attempts - 1:
                    # Last attempt or non-retryable error
                    logger.error(f"Request failed for {self.provider_id} - {endpoint}: {e}")
                    self._track_request_metrics(endpoint=endpoint, start_time=start_time, status="error")
                    
                    # Re-raise appropriate exception
                    if isinstance(e, (ApiError, ConnectionError, TimeoutError)):
                        raise
                    elif isinstance(e, httpx.ConnectError):
                        raise ConnectionError(
                            f"Failed to connect to {self.provider_id}: {e}",
                            provider_id=self.provider_id,
                            endpoint=endpoint,
                        )
                    elif isinstance(e, httpx.TimeoutException):
                        raise TimeoutError(
                            f"Request to {self.provider_id} timed out: {e}",
                            provider_id=self.provider_id,
                            endpoint=endpoint,
                        )
                    else:
                        raise ApiError(
                            f"API request failed: {e}",
                            provider_id=self.provider_id,
                            endpoint=endpoint,
                        )
                
                # Calculate backoff time
                backoff_time = retry_config["initial_backoff"] * (
                    retry_config["backoff_factor"] ** attempt
                )
                backoff_time = min(backoff_time, retry_config["max_backoff"])
                
                logger.warning(
                    f"Retrying request to {self.provider_id} - {endpoint} "
                    f"after {backoff_time:.2f}s (attempt {attempt + 1}/{max_attempts})"
                )
                
                # Wait before retrying
                await asyncio.sleep(backoff_time)
        
        # This should never be reached due to the exception handling above
        raise ApiError(
            f"Request failed after {max_attempts} attempts",
            provider_id=self.provider_id,
            endpoint=endpoint,
        )
    
    @abstractmethod
    def execute(self, request: RequestType, **kwargs: Any) -> ResponseType:
        """
        Execute the tool with the given request.
        
        This method must be implemented by subclasses.
        
        Args:
            request: The request model instance
            **kwargs: Additional keyword arguments
            
        Returns:
            The response data
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def __call__(self, request_data: Dict[str, Any], **kwargs: Any) -> ResponseType:
        """
        Call the tool with the given request data.
        
        This method validates the request data using the request model,
        then calls execute() with the validated request.
        
        Args:
            request_data: The request data as a dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            The response data
        """
        # Validate request data using the request model
        request = self.request_model(**request_data)
        
        # Override provider_id if specified in the request
        if request.provider_id:
            original_provider = self.provider_id
            self.provider_id = request.provider_id
            self.provider_config = self._load_provider_config()
            
            try:
                return self.execute(request, **kwargs)
            finally:
                # Restore original provider
                self.provider_id = original_provider
                self.provider_config = self._load_provider_config()
        
        return self.execute(request, **kwargs)
    
    def close(self) -> None:
        """Close HTTP clients and other resources."""
        self.client.close()
        self.async_client.aclose()
