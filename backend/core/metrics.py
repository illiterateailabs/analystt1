"""
Prometheus Metrics Module

This module defines and manages Prometheus metrics for the application,
providing observability for API usage, costs, performance, and business metrics.
It follows Prometheus naming conventions and best practices.

Key metric categories:
- External API usage (calls, duration, costs)
- Database connection pools
- Cache operations
- Business metrics (analysis tasks, fraud detection)
- System health
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client import REGISTRY, CONTENT_TYPE_LATEST
from prometheus_client.exposition import generate_latest
from starlette.requests import Request
from starlette.responses import Response

# Configure module logger
logger = logging.getLogger(__name__)

# Type variables for generic functions
F = TypeVar('F', bound=Callable[..., Any])
R = TypeVar('R')

# --- External API Metrics ---

# Count of external API calls by provider and endpoint
external_api_calls_total = Counter(
    'external_api_calls_total',
    'Total number of external API calls',
    ['provider', 'endpoint', 'status']
)

# Duration of external API calls
external_api_duration_seconds = Histogram(
    'external_api_duration_seconds',
    'Duration of external API calls in seconds',
    ['provider', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

# Cost of external API calls in USD
external_api_credit_used_total = Counter(
    'external_api_credit_used_total',
    'Total cost of external API calls in USD',
    ['provider', 'credit_type']
)

# Budget usage ratio (0.0-1.0) for external API providers
external_api_budget_ratio = Gauge(
    'external_api_budget_ratio',
    'Ratio of budget used for external API providers (0.0-1.0)',
    ['provider', 'budget_type']  # budget_type: 'daily', 'monthly'
)

# Rate limit remaining for external API providers
external_api_rate_limit_remaining = Gauge(
    'external_api_rate_limit_remaining',
    'Number of API calls remaining before rate limit is hit',
    ['provider', 'limit_type']  # limit_type: 'minute', 'day'
)

# Circuit breaker state for external API providers
external_api_circuit_breaker_state = Gauge(
    'external_api_circuit_breaker_state',
    'Circuit breaker state for external API providers (0=closed, 1=half-open, 2=open)',
    ['provider']
)

# --- LLM Metrics ---

# Count of LLM requests by model
llm_requests_total = Counter(
    'llm_requests_total',
    'Total number of LLM requests',
    ['provider', 'model', 'status']
)

# Duration of LLM requests
llm_request_duration_seconds = Histogram(
    'llm_request_duration_seconds',
    'Duration of LLM requests in seconds',
    ['provider', 'model'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
)

# Token counts for LLM requests
llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total number of tokens processed by LLMs',
    ['provider', 'model', 'token_type']  # token_type: 'input' or 'output'
)

# LLM cost in USD
llm_cost_total = Counter(
    'llm_cost_total',
    'Total cost of LLM usage in USD',
    ['provider', 'model']
)

# --- Agent Execution Metrics ---

# Duration of agent executions
agent_execution_duration_seconds = Histogram(
    'agent_execution_duration_seconds',
    'Duration of agent executions in seconds',
    ['agent_type', 'task_type'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
)

# Count of agent executions
agent_executions_total = Counter(
    'agent_executions_total',
    'Total number of agent executions',
    ['agent_type', 'task_type', 'status']
)

# --- Database Metrics ---

# Database connection pool metrics
db_connections = Gauge(
    'db_connections',
    'Database connection pool status',
    ['database', 'state', 'environment', 'version']  # state: 'used', 'idle', 'max'
)

# Database query metrics
db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Duration of database queries in seconds',
    ['database', 'operation'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# --- Cache Metrics ---

# Cache operation counts
cache_operations_total = Counter(
    'cache_operations_total',
    'Total number of cache operations',
    ['operation', 'status']  # operation: 'get', 'set', 'delete', etc.
)

# Cache hit/miss ratio
cache_hits_total = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['cache_type']  # cache_type: 'data', 'vector', etc.
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['cache_type']
)

# --- Graph Database Metrics ---

# Neo4j query metrics
graph_query_duration_seconds = Histogram(
    'graph_query_duration_seconds',
    'Duration of graph database queries in seconds',
    ['operation'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

# Node and relationship counts
graph_nodes_total = Gauge(
    'graph_nodes_total',
    'Total number of nodes in the graph database',
    ['label']
)

graph_relationships_total = Gauge(
    'graph_relationships_total',
    'Total number of relationships in the graph database',
    ['type']
)

# --- HTTP Metrics ---

# HTTP request duration
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'Duration of HTTP requests in seconds',
    ['method', 'endpoint', 'status_code'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# HTTP request size
http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'Size of HTTP requests in bytes',
    ['method', 'endpoint'],
    buckets=[10, 100, 1000, 10000, 100000, 1000000]
)

# HTTP response size
http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'Size of HTTP responses in bytes',
    ['method', 'endpoint', 'status_code'],
    buckets=[10, 100, 1000, 10000, 100000, 1000000]
)

# --- Business Metrics ---

# Analysis task metrics
analysis_tasks_total = Counter(
    'analysis_tasks_total',
    'Total number of analysis tasks',
    ['analysis_type', 'status', 'environment', 'version']
)

analysis_task_duration_seconds = Histogram(
    'analysis_task_duration_seconds',
    'Duration of analysis tasks in seconds',
    ['analysis_type', 'environment', 'version'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
)

# Fraud detection metrics
fraud_detections_total = Counter(
    'fraud_detections_total',
    'Total number of fraud detections',
    ['detection_type', 'confidence_level', 'environment', 'version']
)

# User activity metrics
user_actions_total = Counter(
    'user_actions_total',
    'Total number of user actions',
    ['action_type', 'environment', 'version']
)

# --- Helper Classes ---

class ApiMetrics:
    """Helper class for tracking external API metrics."""
    
    @staticmethod
    def track_api_call(provider: str, endpoint: str) -> Callable[[F], F]:
        """
        Decorator to track external API calls with Prometheus metrics.
        
        Args:
            provider: The API provider name (e.g., 'gemini', 'sim')
            endpoint: The API endpoint name
            
        Returns:
            Decorated function that tracks metrics
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    external_api_calls_total.labels(
                        provider=provider,
                        endpoint=endpoint,
                        status="success"
                    ).inc()
                    return result
                except Exception as e:
                    external_api_calls_total.labels(
                        provider=provider,
                        endpoint=endpoint,
                        status="error"
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    external_api_duration_seconds.labels(
                        provider=provider,
                        endpoint=endpoint
                    ).observe(duration)
            
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    external_api_calls_total.labels(
                        provider=provider,
                        endpoint=endpoint,
                        status="success"
                    ).inc()
                    return result
                except Exception as e:
                    external_api_calls_total.labels(
                        provider=provider,
                        endpoint=endpoint,
                        status="error"
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    external_api_duration_seconds.labels(
                        provider=provider,
                        endpoint=endpoint
                    ).observe(duration)
            
            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            return cast(F, sync_wrapper)
        
        return decorator
    
    @staticmethod
    def record_api_cost(provider: str, credit_type: str, cost: float) -> None:
        """
        Record the cost of an external API call.
        
        Args:
            provider: The API provider name (e.g., 'gemini', 'sim')
            credit_type: The type of credit used (e.g., 'token', 'request')
            cost: The cost in USD
        """
        if cost <= 0:
            logger.warning(f"Attempted to record non-positive API cost: {cost} for {provider}/{credit_type}")
            return
        
        external_api_credit_used_total.labels(
            provider=provider,
            credit_type=credit_type
        ).inc(cost)
        logger.debug(f"Recorded API cost: ${cost:.6f} for {provider}/{credit_type}")
    
    @staticmethod
    def update_backpressure_metrics(provider_id: str, status_data: Dict[str, Any]) -> None:
        """
        Update backpressure-related metrics for a provider.
        
        Args:
            provider_id: The provider identifier
            status_data: Provider status data from backpressure manager
        """
        # Update budget ratios
        budget = status_data.get("budget", {})
        if budget:
            # Daily budget ratio
            daily_limit = budget.get("daily_limit_usd", 0)
            if daily_limit > 0:
                daily_spent = budget.get("daily_spent_usd", 0)
                daily_ratio = daily_spent / daily_limit
                external_api_budget_ratio.labels(
                    provider=provider_id,
                    budget_type="daily"
                ).set(daily_ratio)
                
                # Log if approaching limit
                if daily_ratio >= 0.8:
                    logger.warning(f"Provider {provider_id} at {daily_ratio:.1%} of daily budget")
            
            # Monthly budget ratio
            monthly_limit = budget.get("monthly_limit_usd", 0)
            if monthly_limit > 0:
                monthly_spent = budget.get("monthly_spent_usd", 0)
                monthly_ratio = monthly_spent / monthly_limit
                external_api_budget_ratio.labels(
                    provider=provider_id,
                    budget_type="monthly"
                ).set(monthly_ratio)
                
                # Log if approaching limit
                if monthly_ratio >= 0.8:
                    logger.warning(f"Provider {provider_id} at {monthly_ratio:.1%} of monthly budget")
        
        # Update rate limit metrics
        requests = status_data.get("requests", {})
        if requests:
            # Per-minute rate limit
            minute_limit = requests.get("minute_limit", 0)
            if minute_limit > 0:
                minute_remaining = minute_limit - requests.get("requests_this_minute", 0)
                external_api_rate_limit_remaining.labels(
                    provider=provider_id,
                    limit_type="minute"
                ).set(minute_remaining)
            
            # Per-day rate limit
            daily_limit = requests.get("daily_limit", 0)
            if daily_limit > 0:
                daily_remaining = daily_limit - requests.get("daily_count", 0)
                external_api_rate_limit_remaining.labels(
                    provider=provider_id,
                    limit_type="day"
                ).set(daily_remaining)
        
        # Update circuit breaker state
        circuit_breaker = status_data.get("circuit_breaker", {})
        if circuit_breaker:
            state = circuit_breaker.get("state", "closed")
            # Convert string state to numeric value for the gauge
            state_value = 0  # closed (default)
            if state == "half_open":
                state_value = 1
            elif state == "open":
                state_value = 2
                
            external_api_circuit_breaker_state.labels(
                provider=provider_id
            ).set(state_value)
            
            # Log circuit breaker state changes
            if state != "closed":
                logger.warning(f"Provider {provider_id} circuit breaker in {state} state")
    
    @staticmethod
    def update_all_providers_metrics(providers_status: Dict[str, Dict[str, Any]]) -> None:
        """
        Update metrics for all providers.
        
        Args:
            providers_status: Dictionary of provider statuses from backpressure manager
        """
        for provider_id, status in providers_status.items():
            ApiMetrics.update_backpressure_metrics(provider_id, status)


class LlmMetrics:
    """Helper class for tracking LLM metrics."""
    
    @staticmethod
    def track_llm_request(provider: str, model: str) -> Callable[[F], F]:
        """
        Decorator to track LLM requests with Prometheus metrics.
        
        Args:
            provider: The LLM provider name (e.g., 'gemini', 'openai')
            model: The model name (e.g., 'gemini-1.5-pro', 'gpt-4')
            
        Returns:
            Decorated function that tracks metrics
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    llm_requests_total.labels(
                        provider=provider,
                        model=model,
                        status="success"
                    ).inc()
                    return result
                except Exception as e:
                    llm_requests_total.labels(
                        provider=provider,
                        model=model,
                        status="error"
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    llm_request_duration_seconds.labels(
                        provider=provider,
                        model=model
                    ).observe(duration)
            
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    llm_requests_total.labels(
                        provider=provider,
                        model=model,
                        status="success"
                    ).inc()
                    return result
                except Exception as e:
                    llm_requests_total.labels(
                        provider=provider,
                        model=model,
                        status="error"
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    llm_request_duration_seconds.labels(
                        provider=provider,
                        model=model
                    ).observe(duration)
            
            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            return cast(F, sync_wrapper)
        
        return decorator
    
    @staticmethod
    def record_token_usage(provider: str, model: str, input_tokens: int, output_tokens: int, cost: Optional[float] = None) -> None:
        """
        Record token usage and optionally cost for an LLM request.
        
        Args:
            provider: The LLM provider name (e.g., 'gemini', 'openai')
            model: The model name (e.g., 'gemini-1.5-pro', 'gpt-4')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Optional cost in USD
        """
        if input_tokens > 0:
            llm_tokens_total.labels(
                provider=provider,
                model=model,
                token_type="input"
            ).inc(input_tokens)
        
        if output_tokens > 0:
            llm_tokens_total.labels(
                provider=provider,
                model=model,
                token_type="output"
            ).inc(output_tokens)
        
        if cost is not None and cost > 0:
            llm_cost_total.labels(
                provider=provider,
                model=model
            ).inc(cost)
            
            # Also record as external API cost for consistency
            external_api_credit_used_total.labels(
                provider=provider,
                credit_type="llm"
            ).inc(cost)


class DatabaseMetrics:
    """Helper class for tracking database metrics."""
    
    @staticmethod
    def set_pool_metrics(database: str, used: int, idle: int, max_size: int, environment: str, version: str) -> None:
        """
        Set database connection pool metrics.
        
        Args:
            database: Database name (e.g., 'postgres', 'neo4j')
            used: Number of used connections
            idle: Number of idle connections
            max_size: Maximum pool size
            environment: Deployment environment
            version: Application version
        """
        db_connections.labels(database=database, state="used", environment=environment, version=version).set(used)
        db_connections.labels(database=database, state="idle", environment=environment, version=version).set(idle)
        db_connections.labels(database=database, state="max", environment=environment, version=version).set(max_size)
    
    @staticmethod
    def track_query(database: str, operation: str) -> Callable[[F], F]:
        """
        Decorator to track database query metrics.
        
        Args:
            database: Database name (e.g., 'postgres', 'neo4j')
            operation: Query operation type (e.g., 'select', 'insert')
            
        Returns:
            Decorated function that tracks metrics
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    db_query_duration_seconds.labels(
                        database=database,
                        operation=operation
                    ).observe(duration)
            
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    db_query_duration_seconds.labels(
                        database=database,
                        operation=operation
                    ).observe(duration)
            
            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            return cast(F, sync_wrapper)
        
        return decorator


class CacheMetrics:
    """Helper class for tracking cache metrics."""
    
    @staticmethod
    def track_operation(operation: str) -> Callable[[F], F]:
        """
        Decorator to track cache operation metrics.
        
        Args:
            operation: Cache operation type (e.g., 'get', 'set')
            
        Returns:
            Decorated function that tracks metrics
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    result = await func(*args, **kwargs)
                    cache_operations_total.labels(
                        operation=operation,
                        status="success"
                    ).inc()
                    return result
                except Exception as e:
                    cache_operations_total.labels(
                        operation=operation,
                        status="error"
                    ).inc()
                    raise
            
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    result = func(*args, **kwargs)
                    cache_operations_total.labels(
                        operation=operation,
                        status="success"
                    ).inc()
                    return result
                except Exception as e:
                    cache_operations_total.labels(
                        operation=operation,
                        status="error"
                    ).inc()
                    raise
            
            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            return cast(F, sync_wrapper)
        
        return decorator
    
    @staticmethod
    def record_cache_hit(cache_type: str) -> None:
        """
        Record a cache hit.
        
        Args:
            cache_type: Type of cache (e.g., 'data', 'vector')
        """
        cache_hits_total.labels(cache_type=cache_type).inc()
    
    @staticmethod
    def record_cache_miss(cache_type: str) -> None:
        """
        Record a cache miss.
        
        Args:
            cache_type: Type of cache (e.g., 'data', 'vector')
        """
        cache_misses_total.labels(cache_type=cache_type).inc()


class BusinessMetrics:
    """Helper class for tracking business metrics."""
    
    @staticmethod
    def track_analysis_task(analysis_type: str, func: Callable[..., R], environment: str = "development", version: str = "unknown") -> Callable[..., R]:
        """
        Decorator to track analysis task metrics.
        
        Args:
            analysis_type: Type of analysis (e.g., 'fraud_detection', 'whale_tracking')
            func: Function to decorate
            environment: Deployment environment
            version: Application version
            
        Returns:
            Decorated function that tracks metrics
        """
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> R:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                analysis_tasks_total.labels(
                    analysis_type=analysis_type,
                    status="success",
                    environment=environment,
                    version=version
                ).inc()
                return result
            except Exception as e:
                analysis_tasks_total.labels(
                    analysis_type=analysis_type,
                    status="error",
                    environment=environment,
                    version=version
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                analysis_task_duration_seconds.labels(
                    analysis_type=analysis_type,
                    environment=environment,
                    version=version
                ).observe(duration)
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> R:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                analysis_tasks_total.labels(
                    analysis_type=analysis_type,
                    status="success",
                    environment=environment,
                    version=version
                ).inc()
                return result
            except Exception as e:
                analysis_tasks_total.labels(
                    analysis_type=analysis_type,
                    status="error",
                    environment=environment,
                    version=version
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                analysis_task_duration_seconds.labels(
                    analysis_type=analysis_type,
                    environment=environment,
                    version=version
                ).observe(duration)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    @staticmethod
    def record_fraud_detection(detection_type: str, confidence_level: str, environment: str = "development", version: str = "unknown") -> None:
        """
        Record a fraud detection.
        
        Args:
            detection_type: Type of fraud detection (e.g., 'whale', 'structuring')
            confidence_level: Confidence level (e.g., 'high', 'medium', 'low')
            environment: Deployment environment
            version: Application version
        """
        fraud_detections_total.labels(
            detection_type=detection_type,
            confidence_level=confidence_level,
            environment=environment,
            version=version
        ).inc()
    
    @staticmethod
    def record_user_action(action_type: str, environment: str = "development", version: str = "unknown") -> None:
        """
        Record a user action.
        
        Args:
            action_type: Type of user action (e.g., 'login', 'search', 'analysis')
            environment: Deployment environment
            version: Application version
        """
        user_actions_total.labels(
            action_type=action_type,
            environment=environment,
            version=version
        ).inc()


# --- FastAPI Integration ---

def setup_metrics(app: FastAPI, environment: str = "development", version: str = "unknown") -> None:
    """
    Set up Prometheus metrics for a FastAPI application.
    
    Args:
        app: FastAPI application instance
        environment: Deployment environment
        version: Application version
    """
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Extract endpoint and method
        method = request.method
        endpoint = request.url.path
        
        # Skip metrics endpoint itself to avoid recursion
        if endpoint == "/metrics":
            return await call_next(request)
        
        # Calculate request size
        request_size = 0
        if "content-length" in request.headers:
            request_size = int(request.headers["content-length"])
        
        http_request_size_bytes.labels(
            method=method,
            endpoint=endpoint
        ).observe(request_size)
        
        # Process request
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        status_code = response.status_code
        
        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).observe(duration)
        
        # Calculate response size
        response_size = 0
        if "content-length" in response.headers:
            response_size = int(response.headers["content-length"])
        
        http_response_size_bytes.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).observe(response_size)
        
        return response
    
    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        """Expose Prometheus metrics."""
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST
        )
    
    logger.info(f"Prometheus metrics set up for {environment}/{version}")


# Import asyncio at the end to avoid circular imports
import asyncio
