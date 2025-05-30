"""Main FastAPI application for the Analyst's Augmentation Agent."""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import sentry_sdk
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlAlchemyIntegration
from starlette.middleware.base import BaseHTTPMiddleware

from backend.config import settings
from backend.core.logging import setup_logging
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.gemini_client import GeminiClient
from backend.integrations.e2b_client import E2BClient


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Sentry if DSN is provided
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        integrations=[
            FastApiIntegration(auto_enabling=True),
            SqlAlchemyIntegration(),
        ],
        traces_sample_rate=0.1,
        environment="development" if settings.debug else "production",
    )


# Middleware to add rate limit headers to responses
class RateLimitHeaderMiddleware(BaseHTTPMiddleware):
    """Middleware to add rate limit headers to responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add rate limit headers if they exist in request state
        if hasattr(request.state, "rate_limit"):
            rate_limit = request.state.rate_limit
            response.headers["X-RateLimit-Limit"] = str(rate_limit["limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_limit["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_limit["reset"])
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting Analyst's Augmentation Agent...")
    
    # Initialize core services
    try:
        # Initialize Neo4j connection
        neo4j_client = Neo4jClient()
        await neo4j_client.connect()
        app.state.neo4j = neo4j_client
        logger.info("Neo4j connection established")
        
        # Initialize Gemini client
        gemini_client = GeminiClient()
        app.state.gemini = gemini_client
        logger.info("Gemini client initialized")
        
        # Initialize e2b client
        e2b_client = E2BClient()
        app.state.e2b = e2b_client
        logger.info("e2b client initialized")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down services...")
    try:
        if hasattr(app.state, 'neo4j'):
            await app.state.neo4j.close()
            logger.info("Neo4j connection closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="An AI-powered system for analyst workflow augmentation",
    lifespan=lifespan,
    debug=settings.debug,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
)

# Add rate limit header middleware
app.add_middleware(RateLimitHeaderMiddleware)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "services": {
            "neo4j": "connected" if hasattr(app.state, 'neo4j') else "disconnected",
            "gemini": "initialized" if hasattr(app.state, 'gemini') else "not_initialized",
            "e2b": "initialized" if hasattr(app.state, 'e2b') else "not_initialized",
        }
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the Analyst's Augmentation Agent",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


# Import and include API routers
from backend.api.v1.chat import router as chat_router
from backend.api.v1.analysis import router as analysis_router
from backend.api.v1.graph import router as graph_router
from backend.api.v1.auth import router as auth_router
from backend.api.v1.crew import router as crew_router

app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(graph_router, prefix="/api/v1/graph", tags=["graph"])
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(crew_router, prefix="/api/v1/crew", tags=["crew"])


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
