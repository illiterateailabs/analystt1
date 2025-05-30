"""
Analyst's Augmentation Agent - FastAPI Application Entry Point

This module serves as the main entry point for the FastAPI application,
configuring middleware, routers, logging, and event handlers.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request, Response, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.config import settings
from backend.core.logging import setup_logging
from backend.api.v1 import auth, chat, crew, graph, analysis
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.e2b_client import E2BClient
from backend.integrations.gemini_client import GeminiClient

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI-powered system for analyst workflows with multimodal understanding, graph analytics, and secure code execution.",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    debug=settings.debug,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global clients
neo4j_client = Neo4jClient()
e2b_client = E2BClient()
gemini_client = GeminiClient()


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize connections and resources on startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Connect to Neo4j
    try:
        await neo4j_client.connect()
        logger.info("Connected to Neo4j database")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
    
    # Initialize other resources
    logger.info("Application startup complete")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down application")
    
    # Close Neo4j connection
    try:
        if hasattr(neo4j_client, 'driver') and neo4j_client.driver is not None:
            await neo4j_client.close()
            logger.info("Closed Neo4j connection")
    except Exception as e:
        logger.error(f"Error closing Neo4j connection: {e}")
    
    # Close any active e2b sandboxes
    try:
        await e2b_client.close_all_sandboxes()
        logger.info("Closed all e2b sandboxes")
    except Exception as e:
        logger.error(f"Error closing e2b sandboxes: {e}")
    
    logger.info("Application shutdown complete")


# Request middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        process_time = time.time() - start_time
        
        # Create a proper error response
        error_response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "type": "server_error",
                "instance": request.url.path,
            }
        )
        error_response.headers["X-Process-Time"] = str(process_time)
        return error_response


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Custom HTTP exception handler with logging."""
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed responses."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body,
            "type": "validation_error",
            "instance": request.url.path,
        },
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring."""
    # Check Neo4j connection
    neo4j_status = "healthy"
    try:
        if not hasattr(neo4j_client, 'driver') or neo4j_client.driver is None:
            await neo4j_client.connect()
        result = await neo4j_client.execute_query("RETURN 1 as n")
        if not result or not result[0].get('n') == 1:
            neo4j_status = "degraded"
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        neo4j_status = "unhealthy"
    
    # Check Gemini API
    gemini_status = "healthy"
    try:
        # Simple ping to Gemini
        await gemini_client.generate_text("Hello")
    except Exception as e:
        logger.error(f"Gemini API health check failed: {e}")
        gemini_status = "unhealthy"
    
    # Determine overall status
    overall_status = "healthy"
    if neo4j_status == "unhealthy" or gemini_status == "unhealthy":
        overall_status = "unhealthy"
    elif neo4j_status == "degraded" or gemini_status == "degraded":
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "version": settings.app_version,
        "services": {
            "neo4j": neo4j_status,
            "gemini": gemini_status,
            "e2b": "healthy"  # We don't check e2b on every health check to avoid unnecessary sandbox creation
        },
        "timestamp": time.time()
    }


# Debug endpoint (only available in debug mode)
@app.get("/debug/config", tags=["Debug"])
async def debug_config():
    """Return application configuration (debug mode only)."""
    if not settings.debug:
        raise StarletteHTTPException(status_code=404, detail="Not found")
    
    # Return non-sensitive configuration
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "cors_origins": settings.cors_origins,
        "neo4j_uri": settings.neo4j_uri,
        "neo4j_username": settings.neo4j_username,
        "neo4j_database": settings.neo4j_database,
        "e2b_template_id": settings.e2b_template_id,
        "gemini_model": settings.gemini_model,
    }


# Include API routers
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["Authentication"]
)

app.include_router(
    chat.router,
    prefix="/api/v1/chat",
    tags=["Chat"]
)

app.include_router(
    crew.router,
    prefix="/api/v1/crew",
    tags=["CrewAI"]
)

app.include_router(
    graph.router,
    prefix="/api/v1/graph",
    tags=["Graph"]
)

app.include_router(
    analysis.router,
    prefix="/api/v1/analysis",
    tags=["Analysis"]
)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "api_prefix": "/api/v1"
    }


# Make the app available to uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
