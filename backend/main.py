"""
Analyst's Augmentation Agent - FastAPI Application Entry Point

This module initializes the FastAPI application, configures middleware,
mounts API routers, and sets up client connections for the Analyst's
Augmentation Agent.
"""

import logging
import os
import time
import subprocess
import datetime
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from fastapi.openapi.utils import get_openapi

# Import configuration
from backend.config import settings, GeminiConfig, Neo4jConfig, E2BConfig, JWTConfig
from backend.core.logging import setup_logging

# Import API routers
from backend.api.v1.analysis import router as analysis_router
from backend.api.v1.auth import router as auth_router
from backend.api.v1.chat import router as chat_router
from backend.api.v1.crew import router as crew_router
from backend.api.v1.graph import router as graph_router
from backend.api.v1.prompts import router as prompts_router

# Import clients
from backend.integrations.neo4j_client import Neo4jClient
from backend.integrations.gemini_client import GeminiClient
from backend.integrations.e2b_client import E2BClient

# Import CrewFactory for health checks
from backend.agents.factory import CrewFactory

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


def get_git_sha() -> str:
    """Get the current git SHA."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()[:7]
    except Exception:
        return "unknown"


def get_build_info() -> Dict[str, str]:
    """Get build information."""
    return {
        "git_sha": get_git_sha(),
        "build_time": datetime.datetime.utcnow().isoformat(),
        "version": settings.app_version,
    }


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Handles startup and shutdown events, including initializing and
    closing connections to external services.
    """
    # Startup: Initialize clients and store in app state
    logger.info("Initializing application clients...")
    
    # Initialize Neo4j client
    neo4j_client = Neo4jClient(
        uri=Neo4jConfig.URI,
        username=Neo4jConfig.USERNAME,
        password=Neo4jConfig.PASSWORD,
        database=Neo4jConfig.DATABASE
    )
    
    # Initialize Gemini client
    gemini_client = GeminiClient(
        api_key=GeminiConfig.API_KEY,
        model=GeminiConfig.MODEL
    )
    
    # Initialize e2b client
    e2b_client = E2BClient(
        api_key=E2BConfig.API_KEY,
        template_id=E2BConfig.TEMPLATE_ID
    )
    
    # Connect to Neo4j
    try:
        await neo4j_client.connect()
        logger.info("Connected to Neo4j database")
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {e}")
        # We'll continue even if Neo4j connection fails, as some endpoints might not need it
    
    # Check if Neo4j connection is required
    if os.getenv("REQUIRE_NEO4J", "false").lower() == "true":
        if not hasattr(neo4j_client, 'driver') or neo4j_client.driver is None:
            logger.error("REQUIRE_NEO4J is true but Neo4j connection failed")
            raise Exception("Neo4j connection required but failed")
    
    # Store clients in app state
    app.state.neo4j = neo4j_client
    app.state.gemini = gemini_client
    app.state.e2b = e2b_client
    
    # Initialize CrewFactory for health checks
    app.state.crew_factory = CrewFactory()
    
    # Store build info in app state
    app.state.build_info = get_build_info()
    
    # Application startup complete
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown: Close connections
    logger.info("Shutting down application...")
    
    # Close Neo4j connection
    if hasattr(app.state, "neo4j"):
        try:
            await app.state.neo4j.close()
            logger.info("Closed Neo4j connection")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {e}")
    
    # Close e2b sandboxes if any are active
    if hasattr(app.state, "e2b"):
        try:
            await app.state.e2b.close_all_sandboxes()
            logger.info("Closed all e2b sandboxes")
        except Exception as e:
            logger.error(f"Error closing e2b sandboxes: {e}")
    
    # Close CrewFactory connections
    if hasattr(app.state, "crew_factory"):
        try:
            await app.state.crew_factory.close()
            logger.info("Closed CrewFactory connections")
        except Exception as e:
            logger.error(f"Error closing CrewFactory connections: {e}")
    
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="API for the Analyst's Augmentation Agent, powered by CrewAI, Gemini, and Neo4j",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    debug=settings.debug
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request information and timing."""
    start_time = time.time()
    
    # Generate request ID
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Log request
    logger.info(
        f"Request started: {request.method} {request.url.path} "
        f"(ID: {request_id})"
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"(ID: {request_id}, Status: {response.status_code}, "
            f"Time: {process_time:.3f}s)"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        return response
    except Exception as e:
        # Log error
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"(ID: {request_id}, Error: {str(e)}, "
            f"Time: {process_time:.3f}s)"
        )
        
        # Return error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "request_id": request_id
            }
        )


# Error handling middleware
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(
        f"HTTP exception: {exc.status_code} {exc.detail} "
        f"(Path: {request.url.path})"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    # Get request ID if available
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.exception(
        f"Unhandled exception in {request.method} {request.url.path} "
        f"(ID: {request_id}): {str(exc)}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "request_id": request_id
        }
    )


# Mount API routers
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(chat_router, prefix="/chat", tags=["Chat"])
api_router.include_router(crew_router, prefix="/crew", tags=["Crew"])
api_router.include_router(graph_router, prefix="/graph", tags=["Graph"])
api_router.include_router(prompts_router, prefix="/prompts", tags=["Prompts"])

app.include_router(api_router)


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "version": settings.app_version,
        "timestamp": time.time(),
        "build_info": app.state.build_info
    }


@app.get("/health/neo4j", tags=["Health"])
async def neo4j_health(neo4j: Neo4jClient = Depends(lambda: app.state.neo4j)):
    """Check Neo4j connection health."""
    try:
        # Test connection with simple query
        result = await neo4j.execute_query("RETURN 1 as test")
        return {
            "status": "ok",
            "connected": True,
            "version": await neo4j.get_server_info(),
            "timestamp": time.time(),
            "build_info": app.state.build_info
        }
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        return {
            "status": "error",
            "connected": False,
            "error": str(e),
            "timestamp": time.time(),
            "build_info": app.state.build_info
        }


@app.get("/health/gemini", tags=["Health"])
async def gemini_health(gemini: GeminiClient = Depends(lambda: app.state.gemini)):
    """Check Gemini API connection health."""
    try:
        # Test connection with simple prompt
        response = await gemini.generate_text("Say hello!")
        return {
            "status": "ok",
            "connected": True,
            "model": GeminiConfig.MODEL,
            "response": response[:50] + "..." if len(response) > 50 else response,
            "timestamp": time.time(),
            "build_info": app.state.build_info
        }
    except Exception as e:
        logger.error(f"Gemini health check failed: {e}")
        return {
            "status": "error",
            "connected": False,
            "error": str(e),
            "timestamp": time.time(),
            "build_info": app.state.build_info
        }


@app.get("/health/crew", tags=["Health"])
async def crew_health(factory: CrewFactory = Depends(lambda: app.state.crew_factory)):
    """Smoke test the CrewFactory."""
    try:
        # Get available crews
        available_crews = factory.get_available_crews()
        
        # Get available tools
        available_tools = list(factory.tools.keys())
        
        return {
            "status": "ok",
            "available_crews": available_crews,
            "available_tools": available_tools,
            "timestamp": time.time(),
            "build_info": app.state.build_info
        }
    except Exception as e:
        logger.error(f"Crew health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
            "build_info": app.state.build_info
        }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Analyst's Augmentation Agent API",
        "docs_url": "/docs",
        "health_check": "/health",
        "build_info": app.state.build_info
    }


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description="API for the Analyst's Augmentation Agent, powered by CrewAI, Gemini, and Neo4j",
        routes=app.routes,
    )
    
    # Add security scheme for JWT
    openapi_schema["components"] = {
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
            }
        }
    }
    
    # Apply security to all endpoints except auth and health
    for path in openapi_schema["paths"]:
        if not any(path.startswith(prefix) for prefix in ["/health", "/api/v1/auth", "/"]):
            for method in openapi_schema["paths"][path]:
                if method.lower() in ["get", "post", "put", "delete", "patch"]:
                    openapi_schema["paths"][path][method]["security"] = [{"bearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Run the application with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run with uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
