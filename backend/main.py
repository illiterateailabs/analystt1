"""
Main FastAPI application for the Analyst Agent.

This module initializes the FastAPI application, sets up middleware,
registers API routes, and configures error handling.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

from backend.config import settings
from backend.core.logging import configure_logging, get_logger
from backend.core.metrics import setup_metrics
from backend.core.events import initialize_events, shutdown_events
from backend.api.v1 import auth, chat, analysis, graph, crew, prompts, webhooks, ws_progress
from backend.integrations.neo4j_client import Neo4jClient

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Analyst Agent API",
    description="API for the Analyst's Augmentation Agent",
    version="0.1.0",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# Set up Prometheus metrics
setup_metrics(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(graph.router, prefix="/api/v1/graph", tags=["Graph"])
app.include_router(crew.router, prefix="/api/v1/crew", tags=["Crew"])
app.include_router(prompts.router, prefix="/api/v1/prompts", tags=["Prompts"])
app.include_router(webhooks.router, prefix="/api/v1/webhooks", tags=["Webhooks"])
app.include_router(ws_progress.router, prefix="/api/v1", tags=["WebSockets"])


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.errors(),
            "status_code": 422,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "details": str(exc) if settings.ENVIRONMENT != "production" else None,
            "status_code": 500,
        },
    )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Health status information
    """
    import platform
    from datetime import datetime
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "python_version": platform.python_version(),
    }


@app.get("/health/neo4j", tags=["Health"])
async def neo4j_health_check(request: Request) -> Dict[str, Any]:
    """
    Neo4j health check endpoint.
    
    Returns:
        Neo4j connection status
    """
    if not settings.REQUIRE_NEO4J:
        return {
            "status": "skipped",
            "message": "Neo4j connection not required in this environment",
        }
    # Re-use the singleton client initialised at startup (if any)
    client: Optional[Neo4jClient] = getattr(request.app.state, "neo4j", None)

    # Fallback: create a transient client if one is not initialised
    if client is None:
        client = Neo4jClient()

    try:
        result = await client.test_connection()
        if result.get("success", False):
            return {
                "status": "connected",
                "version": result.get("version", "unknown"),
                "database": settings.NEO4J_DATABASE,
            }
        else:
            return {
                "status": "error",
                "message": result.get("error", "Unknown error"),
            }
    except Exception as e:
        logger.error(f"Neo4j health check failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
        }


# Startup and shutdown events
@app.on_event("startup")
async def startup_event() -> None:
    """Run startup tasks."""
    logger.info(f"Starting Analyst Agent API ({settings.ENVIRONMENT})")
    
    # Initialize event system
    await initialize_events()
    
    # Initialise and share Neo4j client
    if settings.REQUIRE_NEO4J:
        try:
            neo4j_client = Neo4jClient()
            await neo4j_client.connect()
            app.state.neo4j = neo4j_client  # for dependency injection
            logger.info("Neo4j client initialised and stored in app state")
        except Exception as e:
            logger.error(f"Failed to initialise Neo4j client: {e}")
            if settings.ENVIRONMENT == "production":
                raise  # Hard-fail in prod so misconfiguration is obvious


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Run shutdown tasks."""
    logger.info("Shutting down Analyst Agent API")
    
    # Shutdown event system
    await shutdown_events()
    
    # Close Neo4j connection if initialised
    client: Optional[Neo4jClient] = getattr(app.state, "neo4j", None)
    if client and client.is_connected:
        try:
            await client.close()
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {e}")


# Run the application if executed directly
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
    )
