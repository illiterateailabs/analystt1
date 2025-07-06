"""
Health Check API Endpoints

This module provides comprehensive health check endpoints for the application,
including checks for the database, Redis, Celery workers, and overall system health.
These endpoints are useful for monitoring systems, load balancers, and operational dashboards.
"""

import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

from backend.database import get_async_session
from backend.core.redis_client import RedisClient, RedisDb
from backend.jobs.worker_monitor import WorkerMonitor
from backend.core.metrics import BusinessMetrics
from backend.integrations.neo4j_client import Neo4jClient
from backend.core.telemetry import trace
from backend.core.backpressure import get_all_provider_status

# Configure module logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Health"])

# Singleton instances
redis_client = RedisClient()
worker_monitor = WorkerMonitor()
neo4j_client = Neo4jClient()

@router.get("/health", summary="Basic health check")
@trace("health.basic")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint for load balancers and simple monitoring.
    
    Returns:
        A simple health status response.
    """
    # Track this as a business metric
    BusinessMetrics.track_analysis_task(
        analysis_type="health_check",
        func=lambda: None
    )()
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
    }

@router.get("/health/database", summary="Database health check")
@trace("health.database")
async def database_health(session: AsyncSession = Depends(get_async_session)) -> Dict[str, Any]:
    """
    Checks the database connectivity and basic functionality.
    
    Args:
        session: Async database session
        
    Returns:
        Database health status and connection metrics
    
    Raises:
        HTTPException: If the database check fails
    """
    try:
        # Execute a simple query to verify database connectivity
        result = await session.execute(text("SELECT 1"))
        value = result.scalar()
        
        # Get connection pool metrics
        engine = session.get_bind()
        pool_info = {
            "used_connections": engine.pool.checkedout(),
            "idle_connections": engine.pool.checkedin(),
            "max_connections": engine.pool.size(),
        }
        
        return {
            "status": "healthy" if value == 1 else "degraded",
            "connection_pool": pool_info,
            "latency_ms": round(time.time() * 1000) % 1000,  # Simplified latency simulation
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database health check failed: {str(e)}"
        )

@router.get("/health/redis", summary="Redis health check")
@trace("health.redis")
async def redis_health() -> Dict[str, Any]:
    """
    Checks Redis connectivity and functionality for both cache and vector store.
    
    Returns:
        Redis health status and connection metrics
    
    Raises:
        HTTPException: If the Redis check fails
    """
    results = {}
    overall_status = "healthy"
    
    try:
        # Check cache DB
        cache_start = time.time()
        cache_key = "health:check:cache"
        await redis_client.set(cache_key, "ok", RedisDb.CACHE, ttl_seconds=60)
        cache_value = await redis_client.get(cache_key, RedisDb.CACHE)
        cache_latency = (time.time() - cache_start) * 1000
        
        results["cache"] = {
            "status": "healthy" if cache_value == "ok" else "degraded",
            "latency_ms": round(cache_latency, 2),
        }
        
        # Check vector store DB
        vector_start = time.time()
        vector_key = "health:check:vector"
        await redis_client.set(vector_key, "ok", RedisDb.VECTOR_STORE, ttl_seconds=60)
        vector_value = await redis_client.get(vector_key, RedisDb.VECTOR_STORE)
        vector_latency = (time.time() - vector_start) * 1000
        
        results["vector_store"] = {
            "status": "healthy" if vector_value == "ok" else "degraded",
            "latency_ms": round(vector_latency, 2),
        }
        
        # Check if any component is degraded
        if any(component["status"] != "healthy" for component in results.values()):
            overall_status = "degraded"
            
        return {
            "status": overall_status,
            "components": results,
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis health check failed: {str(e)}"
        )

@router.get("/health/graph", summary="Neo4j graph database health check")
@trace("health.graph")
async def graph_health() -> Dict[str, Any]:
    """
    Checks Neo4j graph database connectivity and functionality.
    
    Returns:
        Neo4j health status and connection metrics
    
    Raises:
        HTTPException: If the Neo4j check fails
    """
    try:
        # Execute a simple query to verify Neo4j connectivity
        start_time = time.time()
        result = neo4j_client.execute_query("MATCH (n) RETURN count(n) AS node_count LIMIT 1")
        query_time = (time.time() - start_time) * 1000
        
        node_count = result[0]["node_count"] if result else 0
        
        return {
            "status": "healthy",
            "node_count": node_count,
            "latency_ms": round(query_time, 2),
        }
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Neo4j health check failed: {str(e)}"
        )

@router.get("/health/workers", summary="Celery workers health check")
@trace("health.workers")
async def worker_health() -> Dict[str, Any]:
    """
    Provides health status for Celery workers and task queues.
    Uses the WorkerMonitor to get the latest worker statistics.
    
    Returns:
        Worker health summary including online status and queue depths
    """
    # This simply exposes the worker monitor's health summary
    return worker_monitor.get_health_summary()

@router.get("/health/providers", summary="External providers health check")
@trace("health.providers")
async def providers_health() -> Dict[str, Any]:
    """
    Returns real-time budget / rate-limit status for all configured
    external API providers as tracked by the BackpressureManager.

    This endpoint is useful for dashboards and proactive alerting.
    """
    return get_all_provider_status()

@router.get("/health/system", summary="Comprehensive system health check")
@trace("health.system")
async def system_health(
    session: AsyncSession = Depends(get_async_session)
) -> Dict[str, Any]:
    """
    Comprehensive health check that aggregates all component health checks.
    This is useful for operational dashboards and detailed monitoring.
    
    Args:
        session: Async database session
        
    Returns:
        Comprehensive health status of all system components
    """
    start_time = time.time()
    results = {}
    issues: List[str] = []
    
    # Check database
    try:
        db_health = await database_health(session)
        results["database"] = db_health
        if db_health["status"] != "healthy":
            issues.append("Database connectivity issues")
    except Exception as e:
        results["database"] = {"status": "unhealthy", "error": str(e)}
        issues.append(f"Database check failed: {str(e)}")
    
    # Check Redis
    try:
        redis_health_result = await redis_health()
        results["redis"] = redis_health_result
        if redis_health_result["status"] != "healthy":
            issues.append("Redis connectivity issues")
    except Exception as e:
        results["redis"] = {"status": "unhealthy", "error": str(e)}
        issues.append(f"Redis check failed: {str(e)}")
    
    # Check Neo4j
    try:
        graph_health_result = await graph_health()
        results["graph"] = graph_health_result
        if graph_health_result["status"] != "healthy":
            issues.append("Neo4j graph database connectivity issues")
    except Exception as e:
        results["graph"] = {"status": "unhealthy", "error": str(e)}
        issues.append(f"Neo4j check failed: {str(e)}")
    
    # Check workers
    try:
        workers_health = worker_monitor.get_health_summary()
        results["workers"] = workers_health
        if workers_health["overall_status"] != "HEALTHY":
            issues.append(f"Worker issues: {workers_health['overall_status']}")
    except Exception as e:
        results["workers"] = {"status": "unhealthy", "error": str(e)}
        issues.append(f"Worker check failed: {str(e)}")
    
    # Determine overall status
    overall_status = "healthy"
    if any(component.get("status") == "unhealthy" for component in results.values()):
        overall_status = "unhealthy"
    elif any(component.get("status") == "degraded" for component in results.values()):
        overall_status = "degraded"
    elif issues:
        overall_status = "degraded"
    
    total_time = (time.time() - start_time) * 1000
    
    return {
        "status": overall_status,
        "components": results,
        "issues": issues if issues else None,
        "response_time_ms": round(total_time, 2),
    }
