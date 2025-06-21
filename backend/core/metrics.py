"""
Prometheus metrics configuration and instrumentation for the application.

This module provides:
1. Middleware for automatic request tracking
2. Custom metrics for business operations
3. Helper functions for manual instrumentation
4. Metrics endpoint setup for Prometheus scraping
"""

import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute
from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    multiprocess,
    generate_latest,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from starlette.types import ASGIApp

# Define common label sets
COMMON_LABELS = ["environment", "version"]
HTTP_LABELS = ["method", "endpoint", "status_code"]
DB_LABELS = ["database", "operation", "status"]
API_LABELS = ["provider", "endpoint", "status"]
LLM_LABELS = ["model", "operation", "status"]
AGENT_LABELS = ["agent_type", "task", "status"]
FRAUD_LABELS = ["detection_type", "chain", "severity"]

# Buckets for latency histograms (in seconds)
LATENCY_BUCKETS = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 30, 60]

# Define metric collectors
# HTTP metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total count of HTTP requests",
    HTTP_LABELS + COMMON_LABELS,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    HTTP_LABELS + COMMON_LABELS,
    buckets=LATENCY_BUCKETS,
)

http_request_size_bytes = Histogram(
    "http_request_size_bytes",
    "HTTP request size in bytes",
    HTTP_LABELS + COMMON_LABELS,
    buckets=[100, 1_000, 10_000, 100_000, 1_000_000],
)

http_response_size_bytes = Histogram(
    "http_response_size_bytes",
    "HTTP response size in bytes",
    HTTP_LABELS + COMMON_LABELS,
    buckets=[100, 1_000, 10_000, 100_000, 1_000_000],
)

# External API metrics
external_api_calls_total = Counter(
    "external_api_calls_total",
    "Total count of external API calls",
    API_LABELS + COMMON_LABELS,
)

external_api_duration_seconds = Histogram(
    "external_api_duration_seconds",
    "External API call duration in seconds",
    API_LABELS + COMMON_LABELS,
    buckets=LATENCY_BUCKETS,
)

external_api_credit_used_total = Counter(
    "external_api_credit_used_total",
    "Total API credits used by external providers",
    API_LABELS + ["credit_type"] + COMMON_LABELS,
)

# Database metrics
db_operations_total = Counter(
    "db_operations_total",
    "Total count of database operations",
    DB_LABELS + COMMON_LABELS,
)

db_operation_duration_seconds = Histogram(
    "db_operation_duration_seconds",
    "Database operation duration in seconds",
    DB_LABELS + COMMON_LABELS,
    buckets=LATENCY_BUCKETS,
)

db_pool_connections = Gauge(
    "db_pool_connections",
    "Database connection pool metrics",
    ["database", "state"] + COMMON_LABELS,
)

db_errors_total = Counter(
    "db_errors_total",
    "Total count of database errors",
    DB_LABELS + ["error_type"] + COMMON_LABELS,
)

# LLM metrics
llm_requests_total = Counter(
    "llm_requests_total",
    "Total count of LLM requests",
    LLM_LABELS + COMMON_LABELS,
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total count of tokens processed by LLMs",
    LLM_LABELS + ["token_type"] + COMMON_LABELS,
)

llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM request latency in seconds",
    LLM_LABELS + COMMON_LABELS,
    buckets=LATENCY_BUCKETS,
)

llm_cost_total = Counter(
    "llm_cost_total",
    "Total cost of LLM usage in USD",
    LLM_LABELS + COMMON_LABELS,
)

# Agent metrics
agent_executions_total = Counter(
    "agent_executions_total",
    "Total count of agent executions",
    AGENT_LABELS + COMMON_LABELS,
)

agent_execution_duration_seconds = Histogram(
    "agent_execution_duration_seconds",
    "Agent execution duration in seconds",
    AGENT_LABELS + COMMON_LABELS,
    buckets=LATENCY_BUCKETS + [120, 300, 600],
)

agent_memory_usage_bytes = Gauge(
    "agent_memory_usage_bytes",
    "Agent memory usage in bytes",
    AGENT_LABELS + COMMON_LABELS,
)

# Business metrics
fraud_detections_total = Counter(
    "fraud_detections_total",
    "Total count of fraud detections",
    FRAUD_LABELS + COMMON_LABELS,
)

fraud_detection_confidence = Histogram(
    "fraud_detection_confidence",
    "Confidence score of fraud detections",
    FRAUD_LABELS + COMMON_LABELS,
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)

graph_nodes_total = Gauge(
    "graph_nodes_total",
    "Total count of nodes in the graph database",
    ["node_type", "chain"] + COMMON_LABELS,
)

graph_relationships_total = Gauge(
    "graph_relationships_total",
    "Total count of relationships in the graph database",
    ["relationship_type", "chain"] + COMMON_LABELS,
)

analysis_tasks_total = Counter(
    "analysis_tasks_total",
    "Total count of analysis tasks",
    ["analysis_type", "status"] + COMMON_LABELS,
)

analysis_task_duration_seconds = Histogram(
    "analysis_task_duration_seconds",
    "Analysis task duration in seconds",
    ["analysis_type", "status"] + COMMON_LABELS,
    buckets=LATENCY_BUCKETS + [120, 300, 600, 1800, 3600],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatically tracking HTTP request metrics.
    
    Captures request counts, durations, and sizes for all endpoints.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        environment: str = "development",
        version: str = "1.8.0-beta",
    ):
        super().__init__(app)
        self.environment = environment
        self.version = version
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract path template for better grouping in Prometheus
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path) if route else request.url.path
        
        # Common labels
        labels = {
            "method": request.method,
            "endpoint": endpoint,
            "environment": self.environment,
            "version": self.version,
        }
        
        # Track request size
        content_length = request.headers.get("content-length")
        if content_length:
            http_request_size_bytes.labels(**labels, status_code="unknown").observe(int(content_length))
        
        # Track request timing
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Update labels with status code
            labels["status_code"] = str(response.status_code)
            
            # Track response size
            resp_content_length = response.headers.get("content-length")
            if resp_content_length:
                http_response_size_bytes.labels(**labels).observe(int(resp_content_length))
            
            return response
        
        except Exception as exc:
            # Handle exceptions and track as 500 errors
            labels["status_code"] = "500"
            raise exc
        
        finally:
            # Always track request count and duration
            http_requests_total.labels(**labels).inc()
            http_request_duration_seconds.labels(**labels).observe(time.time() - start_time)


class DatabaseMetrics:
    """Helper class for tracking database operations."""
    
    @staticmethod
    def track_operation(
        database: str,
        operation: str,
        func: Callable,
        environment: str = "development",
        version: str = "1.8.0-beta",
    ) -> Callable:
        """
        Decorator for tracking database operations.
        
        Args:
            database: The database name (e.g., "neo4j", "postgres", "redis")
            operation: The operation name (e.g., "query", "insert", "update")
            func: The function to wrap
            environment: The environment name
            version: The application version
            
        Returns:
            Decorated function with metrics tracking
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            labels = {
                "database": database,
                "operation": operation,
                "environment": environment,
                "version": version,
                "status": "success",
            }
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Update status and track error
                labels["status"] = "error"
                db_errors_total.labels(
                    **labels,
                    error_type=e.__class__.__name__,
                ).inc()
                raise
            finally:
                # Track operation count and duration
                db_operations_total.labels(**labels).inc()
                db_operation_duration_seconds.labels(**labels).observe(time.time() - start_time)
        
        return wrapper
    
    @staticmethod
    def set_pool_metrics(
        database: str,
        used: int,
        idle: int,
        max_size: int,
        environment: str = "development",
        version: str = "1.8.0-beta",
    ) -> None:
        """
        Set connection pool metrics for a database.
        
        Args:
            database: The database name
            used: Number of connections in use
            idle: Number of idle connections
            max_size: Maximum pool size
            environment: The environment name
            version: The application version
        """
        common = {"database": database, "environment": environment, "version": version}
        
        db_pool_connections.labels(**common, state="used").set(used)
        db_pool_connections.labels(**common, state="idle").set(idle)
        db_pool_connections.labels(**common, state="max").set(max_size)
        db_pool_connections.labels(**common, state="available").set(max_size - used)


class ApiMetrics:
    """Helper class for tracking external API calls."""
    
    @staticmethod
    def track_call(
        provider: str,
        endpoint: str,
        func: Callable,
        environment: str = "development",
        version: str = "1.8.0-beta",
    ) -> Callable:
        """
        Decorator for tracking external API calls.
        
        Args:
            provider: The API provider name (e.g., "sim", "gemini", "e2b")
            endpoint: The endpoint being called
            func: The function to wrap
            environment: The environment name
            version: The application version
            
        Returns:
            Decorated function with metrics tracking
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            labels = {
                "provider": provider,
                "endpoint": endpoint,
                "environment": environment,
                "version": version,
                "status": "success",
            }
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Update status on error
                labels["status"] = "error"
                raise
            finally:
                # Track call count and duration
                external_api_calls_total.labels(**labels).inc()
                external_api_duration_seconds.labels(**labels).observe(time.time() - start_time)
        
        return wrapper
    
    @staticmethod
    def track_credits(
        provider: str,
        endpoint: str,
        credit_type: str,
        amount: float,
        environment: str = "development",
        version: str = "1.8.0-beta",
        status: str = "success",
    ) -> None:
        """
        Track API credit usage.
        
        Args:
            provider: The API provider name
            endpoint: The endpoint being called
            credit_type: The type of credits (e.g., "tokens", "requests", "compute_units")
            amount: The amount of credits used
            environment: The environment name
            version: The application version
            status: The call status
        """
        external_api_credit_used_total.labels(
            provider=provider,
            endpoint=endpoint,
            credit_type=credit_type,
            environment=environment,
            version=version,
            status=status,
        ).inc(amount)


class LlmMetrics:
    """Helper class for tracking LLM operations."""
    
    @staticmethod
    def track_request(
        model: str,
        operation: str,
        func: Callable,
        environment: str = "development",
        version: str = "1.8.0-beta",
    ) -> Callable:
        """
        Decorator for tracking LLM requests.
        
        Args:
            model: The LLM model name
            operation: The operation type (e.g., "completion", "embedding")
            func: The function to wrap
            environment: The environment name
            version: The application version
            
        Returns:
            Decorated function with metrics tracking
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            labels = {
                "model": model,
                "operation": operation,
                "environment": environment,
                "version": version,
                "status": "success",
            }
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Update status on error
                labels["status"] = "error"
                raise
            finally:
                # Track request count and latency
                llm_requests_total.labels(**labels).inc()
                llm_latency_seconds.labels(**labels).observe(time.time() - start_time)
        
        return wrapper
    
    @staticmethod
    def track_tokens(
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int = 0,
        environment: str = "development",
        version: str = "1.8.0-beta",
        status: str = "success",
    ) -> None:
        """
        Track token usage for an LLM request.
        
        Args:
            model: The LLM model name
            operation: The operation type
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            environment: The environment name
            version: The application version
            status: The request status
        """
        common = {
            "model": model,
            "operation": operation,
            "environment": environment,
            "version": version,
            "status": status,
        }
        
        llm_tokens_total.labels(**common, token_type="input").inc(input_tokens)
        
        if output_tokens:
            llm_tokens_total.labels(**common, token_type="output").inc(output_tokens)
    
    @staticmethod
    def track_cost(
        model: str,
        operation: str,
        cost: float,
        environment: str = "development",
        version: str = "1.8.0-beta",
        status: str = "success",
    ) -> None:
        """
        Track cost for an LLM request.
        
        Args:
            model: The LLM model name
            operation: The operation type
            cost: The cost in USD
            environment: The environment name
            version: The application version
            status: The request status
        """
        llm_cost_total.labels(
            model=model,
            operation=operation,
            environment=environment,
            version=version,
            status=status,
        ).inc(cost)


class AgentMetrics:
    """Helper class for tracking agent and crew operations."""
    
    @staticmethod
    def track_execution(
        agent_type: str,
        task: str,
        func: Callable,
        environment: str = "development",
        version: str = "1.8.0-beta",
    ) -> Callable:
        """
        Decorator for tracking agent executions.
        
        Args:
            agent_type: The agent type (e.g., "analyst", "investigator")
            task: The task being performed
            func: The function to wrap
            environment: The environment name
            version: The application version
            
        Returns:
            Decorated function with metrics tracking
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            labels = {
                "agent_type": agent_type,
                "task": task,
                "environment": environment,
                "version": version,
                "status": "success",
            }
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Update status on error
                labels["status"] = "error"
                raise
            finally:
                # Track execution count and duration
                agent_executions_total.labels(**labels).inc()
                agent_execution_duration_seconds.labels(**labels).observe(time.time() - start_time)
        
        return wrapper
    
    @staticmethod
    def set_memory_usage(
        agent_type: str,
        task: str,
        memory_bytes: float,
        environment: str = "development",
        version: str = "1.8.0-beta",
        status: str = "running",
    ) -> None:
        """
        Set memory usage for an agent.
        
        Args:
            agent_type: The agent type
            task: The task being performed
            memory_bytes: Memory usage in bytes
            environment: The environment name
            version: The application version
            status: The agent status
        """
        agent_memory_usage_bytes.labels(
            agent_type=agent_type,
            task=task,
            environment=environment,
            version=version,
            status=status,
        ).set(memory_bytes)


class BusinessMetrics:
    """Helper class for tracking business-specific metrics."""
    
    @staticmethod
    def track_fraud_detection(
        detection_type: str,
        chain: str,
        severity: str,
        confidence: float,
        environment: str = "development",
        version: str = "1.8.0-beta",
    ) -> None:
        """
        Track a fraud detection event.
        
        Args:
            detection_type: The type of fraud detected
            chain: The blockchain or system where fraud was detected
            severity: The severity level (e.g., "high", "medium", "low")
            confidence: The confidence score (0.0-1.0)
            environment: The environment name
            version: The application version
        """
        labels = {
            "detection_type": detection_type,
            "chain": chain,
            "severity": severity,
            "environment": environment,
            "version": version,
        }
        
        fraud_detections_total.labels(**labels).inc()
        fraud_detection_confidence.labels(**labels).observe(confidence)
    
    @staticmethod
    def set_graph_metrics(
        node_counts: Dict[str, int],
        relationship_counts: Dict[str, int],
        chain: str = "all",
        environment: str = "development",
        version: str = "1.8.0-beta",
    ) -> None:
        """
        Set graph database metrics.
        
        Args:
            node_counts: Dictionary mapping node types to counts
            relationship_counts: Dictionary mapping relationship types to counts
            chain: The blockchain or system
            environment: The environment name
            version: The application version
        """
        common = {"chain": chain, "environment": environment, "version": version}
        
        # Set node counts
        for node_type, count in node_counts.items():
            graph_nodes_total.labels(node_type=node_type, **common).set(count)
        
        # Set relationship counts
        for rel_type, count in relationship_counts.items():
            graph_relationships_total.labels(relationship_type=rel_type, **common).set(count)
    
    @staticmethod
    def track_analysis_task(
        analysis_type: str,
        func: Callable,
        environment: str = "development",
        version: str = "1.8.0-beta",
    ) -> Callable:
        """
        Decorator for tracking analysis tasks.
        
        Args:
            analysis_type: The type of analysis being performed
            func: The function to wrap
            environment: The environment name
            version: The application version
            
        Returns:
            Decorated function with metrics tracking
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            labels = {
                "analysis_type": analysis_type,
                "environment": environment,
                "version": version,
                "status": "success",
            }
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Update status on error
                labels["status"] = "error"
                raise
            finally:
                # Track task count and duration
                analysis_tasks_total.labels(**labels).inc()
                analysis_task_duration_seconds.labels(**labels).observe(time.time() - start_time)
        
        return wrapper


def metrics_endpoint() -> StarletteResponse:
    """
    Generate Prometheus metrics response.
    
    Returns:
        Starlette Response with Prometheus metrics in text format
    """
    return StarletteResponse(
        generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


def setup_metrics(app: FastAPI, environment: str = "development", version: str = "1.8.0-beta") -> None:
    """
    Set up metrics for a FastAPI application.
    
    Args:
        app: The FastAPI application
        environment: The environment name
        version: The application version
    """
    # Add metrics endpoint
    app.add_route("/metrics", metrics_endpoint)
    
    # Add metrics middleware
    app.add_middleware(
        MetricsMiddleware,
        environment=environment,
        version=version,
    )
    
    # Initialize multiprocess mode if needed
    # This is useful when running with Gunicorn
    try:
        if "prometheus_multiproc_dir" in app.state.__dict__:
            multiprocess.MultiProcessCollector(REGISTRY)
    except Exception:
        pass
