"""
OpenTelemetry (OTEL) Configuration and Instrumentation.

This module provides a centralized setup for OpenTelemetry, enabling distributed
tracing for the application. It handles:
1.  Initialization of the OTEL SDK.
2.  Configuration of the OTLP exporter to send traces to a collector (e.g., Grafana Tempo).
3.  Automatic instrumentation of key libraries like FastAPI, SQLAlchemy, and Redis.
4.  A custom decorator (`@trace`) for creating fine-grained, application-specific spans.
"""

import functools
import logging
import os
import asyncio
from typing import Any, Callable

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from backend.database import get_engine

logger = logging.getLogger(__name__)

# Get a tracer instance for custom instrumentation
tracer = trace.get_tracer(__name__)


def setup_telemetry(app: FastAPI) -> None:
    """
    Configures OpenTelemetry instrumentation for the FastAPI application.

    This function should be called once on application startup.

    Args:
        app: The FastAPI application instance.
    """
    # Check if telemetry is enabled via an environment variable
    if os.getenv("OTEL_ENABLED", "false").lower() != "true":
        logger.info("OpenTelemetry is disabled. Skipping setup.")
        return

    # Read configuration from environment variables
    service_name = os.getenv("OTEL_SERVICE_NAME", "analyst-droid-one")
    service_version = os.getenv("APP_VERSION", "unknown")
    environment = os.getenv("ENVIRONMENT", "development")
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    if not otlp_endpoint:
        logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT is not set. Telemetry will not be exported.")
        return

    # Create a resource to identify the service
    resource = Resource(attributes={
        "service.name": service_name,
        "service.version": service_version,
        "deployment.environment": environment,
    })

    # Set up the OTLP exporter
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)

    # Set up a tracer provider with the resource and exporter
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    logger.info(f"OpenTelemetry configured to export to '{otlp_endpoint}'")

    # --- Automatic Instrumentation ---

    # Instrument FastAPI for incoming requests
    FastAPIInstrumentor.instrument_app(app)
    logger.debug("FastAPI instrumentation enabled.")

    # Instrument SQLAlchemy for database calls
    try:
        engine = get_engine()
        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
        logger.debug("SQLAlchemy instrumentation enabled.")
    except Exception as e:
        logger.error(f"Failed to instrument SQLAlchemy: {e}")

    # Instrument Redis for cache and vector store operations
    try:
        RedisInstrumentor().instrument()
        logger.debug("Redis instrumentation enabled.")
    except Exception as e:
        logger.error(f"Failed to instrument Redis: {e}")

    logger.info("OpenTelemetry setup complete. Tracing is active.")


def trace(name: str = None) -> Callable:
    """
    A decorator to create a custom OpenTelemetry span around a function.

    This is useful for tracing specific business logic or performance-critical
    sections of the code that are not covered by automatic instrumentation.

    Usage:
        @trace()
        async def my_function():
            ...

        @trace("custom.span.name")
        def another_function():
            ...

    Args:
        name (str, optional): The name for the span. If not provided, the
                              function's qualified name is used.

    Returns:
        The decorated function.
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__qualname__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                # Add function arguments as span attributes for context
                for i, arg in enumerate(args):
                    span.set_attribute(f"arg.{i}", str(arg))
                for k, v in kwargs.items():
                    span.set_attribute(f"kwarg.{k}", str(v))

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, f"Exception: {e}"))
                    span.record_exception(e)
                    logger.error(f"Exception in traced function '{span_name}': {e}", exc_info=True)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                # Add function arguments as span attributes
                for i, arg in enumerate(args):
                    span.set_attribute(f"arg.{i}", str(arg))
                for k, v in kwargs.items():
                    span.set_attribute(f"kwarg.{k}", str(v))

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, f"Exception: {e}"))
                    span.record_exception(e)
                    logger.error(f"Exception in traced function '{span_name}': {e}", exc_info=True)
                    raise

        # Return the appropriate wrapper based on whether the decorated function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Example of how to use the custom tracer directly for more control
async def manual_span_example():
    """An example of creating spans manually without the decorator."""
    with tracer.start_as_current_span("parent.span") as parent_span:
        parent_span.set_attribute("example.id", "123")
        logger.info("Inside the parent span.")

        with tracer.start_as_current_span("child.span") as child_span:
            child_span.set_attribute("child.task", "processing")
            await asyncio.sleep(0.1)
            child_span.add_event("Finished processing", {"items": 10})
            logger.info("Inside the child span.")

        logger.info("Finished child span, back in parent.")
    logger.info("Finished parent span.")
