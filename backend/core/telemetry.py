"""
OpenTelemetry Integration for Coding Analyst Droid

This module sets up OpenTelemetry for distributed tracing, metrics, and logging.
It provides:
1. Automatic instrumentation for FastAPI.
2. Custom spans for CrewAI agent operations.
3. Tracing for API calls, database operations, and Redis interactions.
4. Custom attributes for blockchain-specific data.
5. Support for distributed tracing across services.
6. Integration with existing metrics and logging.
7. Span decorators for easy instrumentation.
8. Error and exception tracking.
9. Configurable exporters (Jaeger, OTLP, console).
10. Configuration via environment variables.
"""

import asyncio
import functools
import logging
import os
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

from fastapi import FastAPI, Request, Response
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.w3c.trace_context_propagator import (
    TraceContextTextMapPropagator,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.semconv.trace import SpanAttributes

# Configure module logger
logger = logging.getLogger(__name__)

# Global tracer instance
tracer = trace.get_tracer(__name__)


def setup_telemetry(app: FastAPI) -> None:
    """
    Sets up OpenTelemetry for the FastAPI application.

    Args:
        app: The FastAPI application instance.
    """
    service_name = os.getenv("OTEL_SERVICE_NAME", "coding-analyst-droid")
    otel_exporter_type = os.getenv("OTEL_EXPORTER_TYPE", "console").lower()
    otel_exporter_endpoint = os.getenv("OTEL_EXPORTER_ENDPOINT")
    otel_debug = os.getenv("OTEL_DEBUG", "false").lower() == "true"

    # Create resource with service name
    resource = Resource.create({SERVICE_NAME: service_name})
    
    # Set up tracer provider
    provider = TracerProvider(resource=resource)

    # Configure exporter based on environment variables
    exporter: SpanExporter
    if otel_exporter_type == "jaeger":
        if not otel_exporter_endpoint:
            logger.warning("OTEL_EXPORTER_ENDPOINT not set for Jaeger exporter. Defaulting to console.")
            exporter = ConsoleSpanExporter()
        else:
            host, port = otel_exporter_endpoint.split(":")
            exporter = JaegerExporter(agent_host_name=host, agent_port=int(port))
            logger.info(f"Using Jaeger exporter at {otel_exporter_endpoint}")
    elif otel_exporter_type == "otlp":
        if not otel_exporter_endpoint:
            logger.warning("OTEL_EXPORTER_ENDPOINT not set for OTLP exporter. Defaulting to console.")
            exporter = ConsoleSpanExporter()
        else:
            exporter = OTLPSpanExporter(endpoint=otel_exporter_endpoint)
            logger.info(f"Using OTLP exporter at {otel_exporter_endpoint}")
    else:  # Default to console
        exporter = ConsoleSpanExporter()
        logger.info("Using Console exporter")

    # Use different processors based on debug mode
    if otel_debug:
        # SimpleSpanProcessor exports spans immediately, good for debugging
        processor = SimpleSpanProcessor(exporter)
    else:
        # BatchSpanProcessor batches spans before export, better for production
        processor = BatchSpanProcessor(exporter)
    
    provider.add_span_processor(processor)

    # Set the global tracer provider
    trace.set_tracer_provider(provider)

    # Set the global text map propagator for distributed tracing
    set_global_textmap(TraceContextTextMapPropagator())

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    logger.info("FastAPI instrumentation enabled.")

    # Instrument HTTPX for API calls
    HTTPXClientInstrumentor().instrument()
    logger.info("HTTPX instrumentation enabled.")

    # Instrument Redis
    try:
        RedisInstrumentor().instrument()
        logger.info("Redis instrumentation enabled.")
    except Exception as e:
        logger.warning(f"Failed to instrument Redis: {e}")

    # Instrument logging
    LoggingInstrumentor().instrument(set_logging_format=True)
    
    # Set up custom logging format with trace context
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] "
            "[trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] %(message)s"
        )
    )
    logging.getLogger().handlers = [handler]
    logging.getLogger().setLevel(logging.INFO)  # Ensure logs are captured
    
    logger.info("Logging instrumentation enabled.")
    logger.info("OpenTelemetry setup complete.")


def trace_function(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to trace a synchronous function with OpenTelemetry.

    Args:
        span_name: The name of the span. If None, uses function name.
        attributes: A dictionary of attributes to add to the span.

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = span_name or func.__name__
            with tracer.start_as_current_span(name) as span:
                if attributes:
                    span.set_attributes(attributes)
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def trace_async_function(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to trace an asynchronous function with OpenTelemetry.

    Args:
        span_name: The name of the span. If None, uses function name.
        attributes: A dictionary of attributes to add to the span.

    Returns:
        Decorated async function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = span_name or func.__name__
            with tracer.start_as_current_span(name) as span:
                if attributes:
                    span.set_attributes(attributes)
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


@contextmanager
def trace_operation(
    operation_name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Context manager to trace operations.

    Args:
        operation_name: The name of the operation.
        attributes: Additional attributes to add to the span.
    """
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            span.set_attributes(attributes)
        try:
            yield span
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@asynccontextmanager
async def trace_async_operation(
    operation_name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Async context manager to trace operations.

    Args:
        operation_name: The name of the operation.
        attributes: Additional attributes to add to the span.
    """
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            span.set_attributes(attributes)
        try:
            yield span
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@asynccontextmanager
async def trace_crewai_operation(
    operation_name: str,
    agent_id: Optional[str] = None,
    task_id: Optional[str] = None,
    crew_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Context manager to trace CrewAI agent operations.

    Args:
        operation_name: The name of the CrewAI operation (e.g., "agent_thought", "tool_execution").
        agent_id: The ID of the agent performing the operation.
        task_id: The ID of the task being executed.
        crew_id: The ID of the crew.
        attributes: Additional attributes to add to the span.
    """
    span_name = f"crewai.{operation_name}"
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("crewai.operation", operation_name)
        if agent_id:
            span.set_attribute("crewai.agent.id", agent_id)
        if task_id:
            span.set_attribute("crewai.task.id", task_id)
        if crew_id:
            span.set_attribute("crewai.crew.id", crew_id)
        if attributes:
            span.set_attributes(attributes)
        try:
            yield span
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def add_blockchain_attributes(
    span: trace.Span,
    chain_id: Optional[str] = None,
    address: Optional[str] = None,
    transaction_hash: Optional[str] = None,
    block_number: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """
    Adds blockchain-specific attributes to a given span.

    Args:
        span: The OpenTelemetry span to add attributes to.
        chain_id: The ID of the blockchain network.
        address: A blockchain address.
        transaction_hash: A transaction hash.
        block_number: A block number.
        **kwargs: Additional custom attributes.
    """
    if chain_id:
        span.set_attribute("blockchain.chain_id", chain_id)
    if address:
        span.set_attribute("blockchain.address", address)
    if transaction_hash:
        span.set_attribute("blockchain.transaction_hash", transaction_hash)
    if block_number:
        span.set_attribute("blockchain.block_number", block_number)
    for key, value in kwargs.items():
        span.set_attribute(f"blockchain.{key}", value)


def get_current_span_context() -> Dict[str, str]:
    """
    Retrieves the current span's trace and span IDs.
    Useful for injecting into logs or custom metrics.

    Returns:
        Dictionary with trace_id and span_id
    """
    span_context = trace.get_current_span().get_span_context()
    return {
        "trace_id": format(span_context.trace_id, "032x"),
        "span_id": format(span_context.span_id, "016x"),
    }


def inject_trace_context(carrier: Dict[str, str]) -> None:
    """
    Injects the current trace context into a carrier dictionary.
    Useful for propagating context across service boundaries.

    Args:
        carrier: Dictionary to inject trace context into
    """
    TraceContextTextMapPropagator().inject(carrier)


def extract_trace_context(carrier: Dict[str, str]) -> trace.SpanContext:
    """
    Extracts trace context from a carrier dictionary.
    Useful for continuing traces across service boundaries.

    Args:
        carrier: Dictionary containing trace context

    Returns:
        Extracted SpanContext
    """
    return TraceContextTextMapPropagator().extract(carrier)


# Neo4j specific tracing
def trace_neo4j_query(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    database: Optional[str] = None,
) -> Callable:
    """
    Decorator to trace Neo4j queries.

    Args:
        query: The Cypher query.
        parameters: Query parameters.
        database: Database name.

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span("neo4j.query") as span:
                span.set_attribute("db.system", "neo4j")
                span.set_attribute("db.statement", query)
                if database:
                    span.set_attribute("db.name", database)
                if parameters:
                    # Only log non-sensitive parameters
                    safe_params = {k: v for k, v in parameters.items() if not _is_sensitive_param(k)}
                    span.set_attribute("db.parameters", str(safe_params))
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span("neo4j.query") as span:
                span.set_attribute("db.system", "neo4j")
                span.set_attribute("db.statement", query)
                if database:
                    span.set_attribute("db.name", database)
                if parameters:
                    # Only log non-sensitive parameters
                    safe_params = {k: v for k, v in parameters.items() if not _is_sensitive_param(k)}
                    span.set_attribute("db.parameters", str(safe_params))
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        # Return appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Redis specific tracing
def trace_redis_operation(
    operation: str,
    key: Optional[str] = None,
    db: Optional[int] = None,
) -> Callable:
    """
    Decorator to trace Redis operations.

    Args:
        operation: The Redis operation (e.g., "GET", "SET").
        key: The Redis key.
        db: Redis database number.

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span("redis.operation") as span:
                span.set_attribute("db.system", "redis")
                span.set_attribute("redis.operation", operation)
                if key:
                    span.set_attribute("redis.key", key)
                if db is not None:
                    span.set_attribute("redis.db", db)
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span("redis.operation") as span:
                span.set_attribute("db.system", "redis")
                span.set_attribute("redis.operation", operation)
                if key:
                    span.set_attribute("redis.key", key)
                if db is not None:
                    span.set_attribute("redis.db", db)
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        # Return appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Middleware for FastAPI request tracing
async def telemetry_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to add custom attributes to request spans.
    
    Args:
        request: FastAPI request
        call_next: Next middleware in chain
        
    Returns:
        FastAPI response
    """
    span = trace.get_current_span()
    
    # Add custom attributes to the span
    span.set_attribute("http.route", request.url.path)
    span.set_attribute("http.client_ip", request.client.host if request.client else "unknown")
    
    # Extract user info if available
    user = getattr(request.state, "user", None)
    if user:
        span.set_attribute("user.id", str(user.id))
    
    # Continue processing the request
    response = await call_next(request)
    
    # Add response attributes
    span.set_attribute("http.status_code", response.status_code)
    
    return response


def _is_sensitive_param(param_name: str) -> bool:
    """
    Check if a parameter name suggests it contains sensitive information.
    
    Args:
        param_name: Parameter name to check
        
    Returns:
        True if parameter might contain sensitive information
    """
    sensitive_keywords = ["password", "token", "secret", "key", "auth", "cred"]
    return any(keyword in param_name.lower() for keyword in sensitive_keywords)


# Example usage
@trace_function(span_name="my_sync_operation", attributes={"component": "example"})
def my_sync_function(data: str) -> str:
    """Example synchronous function with tracing."""
    logger.info(f"Executing sync function with data: {data}")
    return f"Processed {data}"


@trace_async_function(span_name="my_async_operation", attributes={"component": "example"})
async def my_async_function(data: str) -> str:
    """Example asynchronous function with tracing."""
    logger.info(f"Executing async function with data: {data}")
    await asyncio.sleep(0.01)  # Simulate async work
    return f"Processed {data} asynchronously"


# Example of how to use the CrewAI context manager
async def run_crew_task():
    """Example of running a CrewAI task with tracing."""
    async with trace_crewai_operation(
        "task_execution",
        agent_id="data_analyst",
        task_id="gather_data",
        crew_id="fraud_detection",
        attributes={"data_source": "sim_api"},
    ) as span:
        logger.info("CrewAI task is running...")
        await asyncio.sleep(0.05)  # Simulate task execution
        
        # Add blockchain-specific attributes
        add_blockchain_attributes(
            span,
            chain_id="ethereum",
            address="0x123abc",
            transaction_hash="0xdef456",
        )
        
        logger.info("CrewAI task completed.")
