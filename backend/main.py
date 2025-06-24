"""
Main FastAPI application entry point.

This module initializes the FastAPI application, sets up routers,
middleware, database connections, and monitoring systems.
"""

import logging
import os
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

# Back-pressure / budget-control middleware
from backend.core.backpressure import BackpressureMiddleware
from backend.api.v1 import (
    analysis,
    auth,
    chat,
    crew,
    graph,
    prompts,
    templates,
    webhooks,
    whale_endpoints,
    ws_progress,
    health,  # new health endpoints (worker / db / redis checks)
)
from backend.auth.dependencies import get_current_user
from backend.core import events, logging as app_logging, metrics, sentry_config
from backend.core.telemetry import setup_telemetry
from backend.database import create_db_and_tables, get_engine
from backend.jobs import sim_graph_job
from backend.jobs.worker_monitor import WorkerMonitor  # start Celery worker monitor

# Configure logging
app_logging.setup_logging()
logger = logging.getLogger(__name__)

# Get environment variables
FASTAPI_DEBUG = os.getenv("FASTAPI_DEBUG", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
APP_VERSION = os.getenv("APP_VERSION", "1.8.0-beta")
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")

# Create FastAPI app
app = FastAPI(
    title="Coding Analyst Droid",
    description="Blockchain fraud analysis platform with agent-based intelligence",
    version=APP_VERSION,
    debug=FASTAPI_DEBUG,
    docs_url="/api/docs" if FASTAPI_DEBUG else None,
    redoc_url="/api/redoc" if FASTAPI_DEBUG else None,
)

# Initialize OpenTelemetry telemetry
@app.on_event("startup")
async def initialize_telemetry() -> None:
    """Configure OpenTelemetry tracing on startup (if OTEL_ENABLED=true)."""
    setup_telemetry(app)
    logger.info("OpenTelemetry telemetry setup executed")

# Initialize Sentry for error tracking
@app.on_event("startup")
async def initialize_sentry() -> None:
    """Initialize Sentry error tracking on startup."""
    sentry_config.init_sentry()
    logger.info("Sentry initialized")

# Set up Prometheus metrics if enabled
if ENABLE_METRICS:
    metrics.setup_metrics(app, environment=ENVIRONMENT, version=APP_VERSION)
    logger.info("Prometheus metrics enabled at /metrics endpoint")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if FASTAPI_DEBUG else [
        "https://analyst-droid.example.com",
        "https://api.analyst-droid.example.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    max_age=3600,  # 1 hour
)

# Mount back-pressure middleware (rate-limit, budget & circuit-breaker)
app.add_middleware(BackpressureMiddleware)

# Include API routers
api_v1 = FastAPI(openapi_prefix="/api/v1")
api_v1.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_v1.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
api_v1.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_v1.include_router(crew.router, prefix="/crew", tags=["Crew"])
api_v1.include_router(graph.router, prefix="/graph", tags=["Graph"])
api_v1.include_router(prompts.router, prefix="/prompts", tags=["Prompts"])
api_v1.include_router(templates.router, prefix="/templates", tags=["Templates"])
api_v1.include_router(webhooks.router, prefix="/webhooks", tags=["Webhooks"])
api_v1.include_router(whale_endpoints.router, prefix="/whales", tags=["Whales"])
# ws_progress router already defines its own `/ws/...` paths – no extra prefix
api_v1.include_router(ws_progress.router, tags=["WebSockets"])
# Health endpoints (detailed component checks)
api_v1.include_router(health.router)  # endpoints define their own /health/* paths

# Mount API v1 under /api/v1
app.mount("/api/v1", api_v1)

# Mount static files if in production
if not FASTAPI_DEBUG:
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for the application.
    
    Args:
        request: The request that caused the exception
        exc: The exception that was raised
        
    Returns:
        JSON response with error details
    """
    logger.exception(f"Unhandled exception: {exc}")
    sentry_config.capture_exception(exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# Database initialization
@app.on_event("startup")
async def initialize_database() -> None:
    """Initialize database connections and tables on startup."""
    try:
        create_db_and_tables()
        logger.info("Database initialized successfully")
        
        # Track database pool metrics
        engine = get_engine()
        metrics.DatabaseMetrics.set_pool_metrics(
            database="postgres",
            used=engine.pool.checkedout(),
            idle=engine.pool.checkedin(),
            max_size=engine.pool.size(),
            environment=ENVIRONMENT,
            version=APP_VERSION,
        )
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sentry_config.capture_exception(e)
        raise

# Initialize scheduled jobs
@app.on_event("startup")
async def initialize_jobs() -> None:
    """Initialize scheduled jobs on startup."""
    try:
        # Start SIM graph ingestion job
        sim_graph_job.start()
        # Start Celery worker monitor background task
        WorkerMonitor().start()
        logger.info("Scheduled jobs initialized")
    except Exception as e:
        logger.error(f"Job initialization failed: {e}")
        sentry_config.capture_exception(e)

# Initialize event system
@app.on_event("startup")
async def initialize_events() -> None:
    """Initialize event system on startup."""
    try:
        events.init_event_system()
        logger.info("Event system initialized")
    except Exception as e:
        logger.error(f"Event system initialization failed: {e}")
        sentry_config.capture_exception(e)

# Shutdown handlers
@app.on_event("shutdown")
async def shutdown_jobs() -> None:
    """Shutdown scheduled jobs."""
    try:
        sim_graph_job.stop()
        logger.info("Scheduled jobs stopped")
    except Exception as e:
        logger.error(f"Error stopping jobs: {e}")
        sentry_config.capture_exception(e)

@app.on_event("shutdown")
async def shutdown_database() -> None:
    """Close database connections."""
    try:
        engine = get_engine()
        await engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")
        sentry_config.capture_exception(e)

# Root endpoint
@app.get("/", tags=["Health"])
async def root() -> Dict[str, Any]:
    """
    Root endpoint for health checks.
    
    Returns:
        Basic application information
    """
    return {
        "name": "Coding Analyst Droid",
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "status": "healthy",
    }

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring systems.
    
    Returns:
        Health status of various components
    """
    # Track this as a business metric
    metrics.BusinessMetrics.track_analysis_task(
        analysis_type="health_check",
        func=lambda: None,
        environment=ENVIRONMENT,
        version=APP_VERSION,
    )()
    
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "components": {
            "api": "healthy",
            "database": "healthy",
            "graph": "healthy",
            "cache": "healthy",
            "jobs": "healthy",
        },
    }

# Run the application if executed directly
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=FASTAPI_DEBUG,
        log_level="debug" if FASTAPI_DEBUG else "info",
    )
