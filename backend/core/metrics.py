"""
Prometheus metrics for the Analyst Agent.

This module provides Prometheus metrics for monitoring the application,
including crew execution, LLM usage, and HITL workflow metrics.
"""

import time
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from functools import wraps
from contextlib import contextmanager

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway, start_http_server
from prometheus_client.exposition import generate_latest
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.config import settings
from backend.core.logging import get_logger

# Configure logging
logger = get_logger(__name__)

# Create registry
registry = CollectorRegistry()

# Crew execution metrics
crew_task_duration_seconds = Histogram(
    name="crew_task_duration_seconds",
    documentation="Duration of crew tasks in seconds",
    labelnames=["crew_name", "agent_id", "task_description"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),  # 0.1s to 10min
    registry=registry
)

crew_task_total = Counter(
    name="crew_task_total",
    documentation="Total number of crew tasks",
    labelnames=["crew_name", "agent_id", "status"],
    registry=registry
)

crew_errors_total = Counter(
    name="crew_errors_total",
    documentation="Total number of crew errors",
    labelnames=["crew_name", "agent_id", "error_type"],
    registry=registry
)

# LLM usage metrics
llm_tokens_used_total = Counter(
    name="llm_tokens_used_total",
    documentation="Total number of tokens used by LLM",
    labelnames=["model", "type"],  # type: input, output
    registry=registry
)

llm_cost_usd_total = Counter(
    name="llm_cost_usd_total",
    documentation="Total cost of LLM usage in USD",
    labelnames=["model"],
    registry=registry
)

llm_requests_total = Counter(
    name="llm_requests_total",
    documentation="Total number of LLM requests",
    labelnames=["model", "status"],  # status: success, error
    registry=registry
)

# HITL workflow metrics
hitl_review_duration_seconds = Histogram(
    name="hitl_review_duration_seconds",
    documentation="Duration of human review in seconds",
    labelnames=["review_type", "risk_level"],
    buckets=(60.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0, 14400.0, 28800.0),  # 1min to 8h
    registry=registry
)

hitl_reviews_total = Counter(
    name="hitl_reviews_total",
    documentation="Total number of human reviews",
    labelnames=["review_type", "status", "risk_level"],  # status: approved, rejected, expired
    registry=registry
)

# Additional metrics
webhook_deliveries_total = Counter(
    name="webhook_deliveries_total",
    documentation="Total number of webhook deliveries",
    labelnames=["webhook_type", "status"],  # status: success, error
    registry=registry
)

active_crews = Gauge(
    name="active_crews",
    documentation="Number of currently active crews",
    labelnames=["crew_name"],
    registry=registry
)

# Helper functions for updating metrics

@contextmanager
def track_crew_task_duration(crew_name: str, agent_id: str, task_description: str):
    """
    Context manager to track the duration of a crew task.
    
    Args:
        crew_name: Name of the crew
        agent_id: ID of the agent
        task_description: Description of the task
    """
    start_time = time.time()
    try:
        yield
        # Task completed successfully
        crew_task_total.labels(crew_name=crew_name, agent_id=agent_id, status="success").inc()
    except Exception as e:
        # Task failed
        crew_task_total.labels(crew_name=crew_name, agent_id=agent_id, status="error").inc()
        crew_errors_total.labels(
            crew_name=crew_name, 
            agent_id=agent_id, 
            error_type=type(e).__name__
        ).inc()
        raise
    finally:
        # Record duration regardless of success/failure
        duration = time.time() - start_time
        crew_task_duration_seconds.labels(
            crew_name=crew_name,
            agent_id=agent_id,
            task_description=task_description
        ).observe(duration)


def track_llm_usage(model: str, input_tokens: int, output_tokens: int, cost_usd: float, success: bool = True):
    """
    Track LLM usage metrics.
    
    Args:
        model: Name of the LLM model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost_usd: Cost in USD
        success: Whether the request was successful
    """
    # Track tokens
    llm_tokens_used_total.labels(model=model, type="input").inc(input_tokens)
    llm_tokens_used_total.labels(model=model, type="output").inc(output_tokens)
    
    # Track cost
    llm_cost_usd_total.labels(model=model).inc(cost_usd)
    
    # Track request
    status = "success" if success else "error"
    llm_requests_total.labels(model=model, status=status).inc()


def track_hitl_review(review_type: str, risk_level: str, status: str, duration_seconds: float):
    """
    Track HITL review metrics.
    
    Args:
        review_type: Type of review (e.g., "compliance")
        risk_level: Risk level (e.g., "high", "medium", "low")
        status: Status of the review (e.g., "approved", "rejected", "expired")
        duration_seconds: Duration of the review in seconds
    """
    # Track review count
    hitl_reviews_total.labels(review_type=review_type, status=status, risk_level=risk_level).inc()
    
    # Track review duration
    hitl_review_duration_seconds.labels(review_type=review_type, risk_level=risk_level).observe(duration_seconds)


def track_webhook_delivery(webhook_type: str, success: bool):
    """
    Track webhook delivery metrics.
    
    Args:
        webhook_type: Type of webhook (e.g., "slack", "email", "teams", "custom_url")
        success: Whether the delivery was successful
    """
    status = "success" if success else "error"
    webhook_deliveries_total.labels(webhook_type=webhook_type, status=status).inc()


def increment_active_crews(crew_name: str):
    """
    Increment the number of active crews.
    
    Args:
        crew_name: Name of the crew
    """
    active_crews.labels(crew_name=crew_name).inc()


def decrement_active_crews(crew_name: str):
    """
    Decrement the number of active crews.
    
    Args:
        crew_name: Name of the crew
    """
    active_crews.labels(crew_name=crew_name).dec()


# Decorator for tracking function execution time
def track_execution_time(metric: Histogram, **labels):
    """
    Decorator to track function execution time using a Histogram metric.
    
    Args:
        metric: The Histogram metric to use
        **labels: Labels to apply to the metric
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# FastAPI middleware for tracking HTTP request metrics
class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to track HTTP request metrics."""
    
    def __init__(
        self,
        app: FastAPI,
        app_name: str = "fastapi",
        skip_paths: List[str] = None,
    ):
        super().__init__(app)
        self.app_name = app_name
        self.skip_paths = skip_paths or []
        
        # Create metrics
        self.requests_total = Counter(
            name=f"{app_name}_requests_total",
            documentation="Total number of HTTP requests",
            labelnames=["method", "endpoint", "status_code"],
            registry=registry
        )
        
        self.requests_duration_seconds = Histogram(
            name=f"{app_name}_request_duration_seconds",
            documentation="HTTP request duration in seconds",
            labelnames=["method", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
            registry=registry
        )
        
        self.requests_in_progress = Gauge(
            name=f"{app_name}_requests_in_progress",
            documentation="Number of HTTP requests in progress",
            labelnames=["method", "endpoint"],
            registry=registry
        )
    
    async def dispatch(self, request: Request, call_next):
        # Skip metrics for certain paths
        if any(request.url.path.startswith(path) for path in self.skip_paths):
            return await call_next(request)
        
        # Get endpoint and method
        method = request.method
        endpoint = request.url.path
        
        # Track in-progress requests
        self.requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        
        # Track request duration
        start_time = time.time()
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            status_code = 500
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            self.requests_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
            self.requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            self.requests_in_progress.labels(method=method, endpoint=endpoint).dec()


# Metrics endpoint handler
async def metrics_endpoint(request: Request) -> Response:
    """
    Endpoint to expose Prometheus metrics.
    
    Args:
        request: FastAPI request
        
    Returns:
        Response with metrics in Prometheus format
    """
    return Response(
        content=generate_latest(registry),
        media_type="text/plain"
    )


def setup_metrics(app: FastAPI) -> None:
    """
    Set up metrics for a FastAPI application.
    
    This function adds the Prometheus middleware and metrics endpoint to the app.
    
    Args:
        app: FastAPI application
    """
    # Add Prometheus middleware
    app.add_middleware(
        PrometheusMiddleware,
        app_name="analyst_agent",
        skip_paths=["/metrics", "/health", "/docs", "/redoc", "/openapi.json"]
    )
    
    # Add metrics endpoint
    app.add_route("/metrics", metrics_endpoint)
    
    logger.info("Prometheus metrics initialized")


# For standalone metrics server
def start_metrics_server(port: int = 8000) -> None:
    """
    Start a standalone metrics server.
    
    Args:
        port: Port to listen on
    """
    try:
        start_http_server(port, registry=registry)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")


# Initialize metrics on module import
import asyncio  # Required for track_execution_time decorator
