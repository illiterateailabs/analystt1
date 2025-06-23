"""
Celery Application Configuration

This module defines and configures the Celery application for handling asynchronous
background jobs. It integrates with the application's provider registry for
configuration and sets up observability with OpenTelemetry.

Key Features:
- Redis broker and result backend configured via the central provider registry.
- Secure JSON serialization for tasks and results.
- Task routing to dedicated queues for better workload management.
- Reliability settings (acks_late) to prevent task loss on worker failure.
- Seamless OpenTelemetry integration for distributed tracing across API and workers.
"""

import logging
import os
from celery import Celery
from opentelemetry.instrumentation.celery import CeleryInstrumentor

from backend.providers import get_provider

logger = logging.getLogger(__name__)


def _get_redis_url_for_celery(purpose: str, default_db: int) -> str:
    """
    Constructs a Redis connection URL for Celery from the provider registry.

    Args:
        purpose: A string describing the purpose (e.g., 'broker', 'backend')
                 used for logging.
        default_db: The default Redis database number to use if not specified
                    in the provider registry.

    Returns:
        A full Redis connection URL string.
    """
    provider_config = get_provider("redis")
    if not provider_config:
        logger.error(f"Redis provider config not found for Celery {purpose}. Falling back to localhost.")
        return f"redis://localhost:6379/{default_db}"

    # The provider registry already substitutes env vars for host, port, etc.
    connection_uri = provider_config.get("connection_uri", "redis://localhost:6379")
    
    # Ensure we use a specific DB for Celery to avoid conflicts
    # redis-py/celery expect format: redis://host:port/db_number
    # Remove any existing DB number from the base URI
    base_url = connection_uri
    if '/' in connection_uri.split('://')[1]:
        base_url = connection_uri.rsplit('/', 1)[0]
    
    redis_url = f"{base_url}/{default_db}"

    logger.info(f"Celery {purpose} configured to use Redis at: {redis_url}")
    return redis_url


# --- Celery App Initialization ---
# This is the central Celery application instance.
celery_app = Celery(
    "analyst_droid_jobs",
    broker=_get_redis_url_for_celery("broker", default_db=2),
    backend=_get_redis_url_for_celery("backend", default_db=3),
    include=[
        "backend.jobs.tasks.analysis_tasks",
        "backend.jobs.tasks.data_tasks",
    ],
)


# --- Core Configuration ---
celery_app.conf.update(
    # Serialization: Use JSON for security and interoperability.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone: Standardize on UTC.
    timezone="UTC",
    enable_utc=True,

    # Reliability: Acknowledge tasks after they complete to prevent loss on worker failure.
    # This is crucial for idempotent tasks.
    task_acks_late=True,
    
    # Worker settings: Prefetch 1 task at a time, suitable for long-running jobs.
    worker_prefetch_multiplier=1,

    # Results Backend: Store results for 24 hours.
    result_expires=86400,
)


# --- Task Routing ---
# Directs tasks to specific queues. This allows scaling workers for different
# workloads independently. For example, you can have more workers for the
# high-priority 'analysis' queue.
celery_app.conf.task_routes = {
    "backend.jobs.tasks.data_tasks.*": {"queue": "data_ingestion"},
    "backend.jobs.tasks.analysis_tasks.*": {"queue": "analysis"},
    # Fallback for any other tasks
    "*": {"queue": "default"},
}


# --- Observability Integration ---
@celery_app.on_after_configure.connect
def setup_celery_instrumentation(sender, **kwargs):
    """
    Instruments the Celery application with OpenTelemetry.

    This signal handler is called when the Celery app is configured, which
    happens in both the main application process (when sending tasks) and in
    each worker process. This ensures tracing is enabled everywhere.
    """
    if os.getenv("OTEL_ENABLED", "false").lower() == "true":
        logger.info("Instrumenting Celery with OpenTelemetry...")
        try:
            CeleryInstrumentor().instrument()
            logger.info("Celery OpenTelemetry instrumentation complete.")
        except Exception as e:
            logger.error(f"Failed to instrument Celery with OpenTelemetry: {e}")

