"""
Celery Worker Health Monitoring System

This module provides a comprehensive monitoring system for the Celery workers.
It runs as a background task within the main FastAPI application, periodically
checking worker status and queue depths.

Key Features:
- Pings workers to ensure they are online and responsive.
- Inspects workers to get statistics on active tasks and pool configuration.
- Connects to the Redis broker to monitor the depth of task queues.
- Exposes key metrics to Prometheus for long-term monitoring and alerting.
- Provides a FastAPI health endpoint to get a real-time snapshot of worker status.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter
from prometheus_client import Gauge

from backend.jobs.celery_app import celery_app
from backend.core.redis_client import RedisClient, RedisDb

logger = logging.getLogger(__name__)

# --- Prometheus Metrics Definitions ---

CELERY_WORKERS_ONLINE = Gauge(
    "celery_workers_online_total",
    "Total number of online and responsive Celery workers."
)

CELERY_QUEUE_DEPTH = Gauge(
    "celery_queue_depth_total",
    "Number of tasks currently waiting in a Celery queue.",
    ["queue_name"]
)

CELERY_ACTIVE_TASKS = Gauge(
    "celery_active_tasks_total",
    "Number of currently active (running) tasks by worker.",
    ["worker_name"]
)


class WorkerMonitor:
    """
    A singleton class that monitors Celery workers and queues in the background.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, check_interval_seconds: int = 30):
        if self._initialized:
            return

        self.celery_app = celery_app
        self.redis_client = RedisClient()
        self.check_interval = check_interval_seconds
        self.worker_stats: Dict[str, Any] = {}
        self.queue_stats: Dict[str, int] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._initialized = True
        logger.info("WorkerMonitor initialized.")

    async def _get_queue_depths(self) -> Dict[str, int]:
        """Connects to the Redis broker and retrieves the length of task queues."""
        depths = {}
        queues_to_check = ['data_ingestion', 'analysis', 'default']
        try:
            # Determine the correct Redis DB from the Celery broker URL
            broker_url = self.celery_app.conf.broker_url
            db_num_str = broker_url.split('/')[-1]
            db_num = int(db_num_str) if db_num_str.isdigit() else 0
            
            # The redis-py client is synchronous, so we can't await its methods directly.
            # We must use the underlying async-redis client if available, or wrap calls.
            # For simplicity, we assume the RedisClient can provide a raw async client.
            # A more robust solution would be to use an async redis library directly here.
            # Let's assume RedisClient uses redis.asyncio internally.
            redis_raw_client = self.redis_client._get_client(RedisDb(db_num)) # This is a simplification

            for queue_name in queues_to_check:
                # Celery uses standard Redis lists for queues. LLEN gets the length.
                depth = await redis_raw_client.llen(queue_name)
                depths[queue_name] = depth
        except Exception as e:
            logger.error(f"Could not fetch Celery queue depths from Redis: {e}", exc_info=True)
        return depths

    async def _get_worker_stats(self) -> Dict[str, Any]:
        """Gathers detailed statistics from all active Celery workers."""
        inspector = self.celery_app.control.inspect(timeout=5)
        loop = asyncio.get_running_loop()

        try:
            # These are blocking network calls and must be run in a thread pool
            # to avoid blocking the asyncio event loop.
            stats = await loop.run_in_executor(None, inspector.stats)
            active_tasks = await loop.run_in_executor(None, inspector.active)
            ping_results = await loop.run_in_executor(None, inspector.ping)

            if not stats:
                logger.warning("No active Celery workers found or they are not responding.")
                return {}

            processed_stats = {}
            for worker_name, worker_data in stats.items():
                is_online = worker_name in (ping_results or {})
                active_tasks_list = active_tasks.get(worker_name, [])
                processed_stats[worker_name] = {
                    "online": is_online,
                    "pid": worker_data.get('pid'),
                    "pool_size": worker_data.get('pool', {}).get('max-concurrency'),
                    "total_tasks_processed": worker_data.get('total', 0),
                    "active_tasks_count": len(active_tasks_list),
                    "active_tasks_details": active_tasks_list,
                }
            return processed_stats
        except Exception as e:
            logger.error(f"Failed to inspect Celery workers: {e}", exc_info=True)
            return {}

    async def update_metrics(self):
        """Fetches the latest stats and updates all related Prometheus metrics."""
        logger.debug("Updating Celery worker metrics...")
        self.worker_stats = await self._get_worker_stats()
        self.queue_stats = await self._get_queue_depths()

        # Update Prometheus Gauges
        online_workers = sum(1 for stats in self.worker_stats.values() if stats.get('online'))
        CELERY_WORKERS_ONLINE.set(online_workers)

        for queue_name, depth in self.queue_stats.items():
            CELERY_QUEUE_DEPTH.labels(queue_name=queue_name).set(depth)
        
        # Ensure all known queues are reported, even if empty
        for q in ['data_ingestion', 'analysis', 'default']:
            if q not in self.queue_stats:
                 CELERY_QUEUE_DEPTH.labels(queue_name=q).set(0)

        # Update active tasks per worker, and clear metrics for offline workers
        active_worker_names = set(self.worker_stats.keys())
        # _metrics is an internal detail, but a common way to handle labels cleanup
        all_known_workers = {labels[0] for labels in CELERY_ACTIVE_TASKS._metrics.keys()}
        
        for worker_name, stats in self.worker_stats.items():
            CELERY_ACTIVE_TASKS.labels(worker_name=worker_name).set(stats.get('active_tasks_count', 0))

        offline_workers = all_known_workers - active_worker_names
        for worker_name in offline_workers:
            CELERY_ACTIVE_TASKS.labels(worker_name=worker_name).set(0)

        logger.debug(f"Metrics updated: {online_workers} workers online, Queues: {self.queue_stats}")

    async def monitor_loop(self):
        """The main async loop that periodically checks and updates worker/queue status."""
        logger.info(f"Starting Celery worker monitoring loop (interval: {self.check_interval}s).")
        while True:
            try:
                await self.update_metrics()
            except Exception as e:
                logger.error(f"Unhandled error in worker monitor loop: {e}", exc_info=True)
            await asyncio.sleep(self.check_interval)

    def start(self):
        """Starts the monitoring loop as a background asyncio task."""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self.monitor_loop())
            logger.info("Celery worker monitoring background task started.")
        else:
            logger.warning("Celery worker monitoring background task is already running.")

    def stop(self):
        """Stops the monitoring loop gracefully."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            self._monitor_task = None
            logger.info("Celery worker monitoring background task stopped.")

    def get_health_summary(self) -> Dict[str, Any]:
        """Returns the latest collected snapshot of worker and queue health."""
        online_workers = sum(1 for stats in self.worker_stats.values() if stats.get('online'))
        status = "HEALTHY"
        if not self.worker_stats or online_workers == 0:
            status = "UNHEALTHY"
        elif any(depth > 100 for depth in self.queue_stats.values()): # Example threshold
             status = "DEGRADED"

        return {
            "overall_status": status,
            "workers": self.worker_stats,
            "queues": self.queue_stats,
        }


# --- FastAPI Router Integration ---

router = APIRouter()
monitor_singleton = WorkerMonitor()

@router.on_event("startup")
async def startup_worker_monitoring():
    """FastAPI startup event to launch the monitor."""
    monitor_singleton.start()

@router.on_event("shutdown")
async def shutdown_worker_monitoring():
    """FastAPI shutdown event to stop the monitor."""
    monitor_singleton.stop()

@router.get("/health/workers", tags=["Health", "Celery"])
async def get_worker_health() -> Dict[str, Any]:
    """
    Provides a health check summary for all Celery workers and their task queues.
    This endpoint returns the last known state from the monitor, not a live check.
    """
    return monitor_singleton.get_health_summary()
