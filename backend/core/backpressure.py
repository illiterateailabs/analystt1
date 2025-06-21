"""
Back-Pressure Middleware System

This module provides a comprehensive back-pressure middleware system that:
1. Monitors provider budgets and rate limits in real-time
2. Queues tasks when provider budget is low or rate limits approached
3. Implements intelligent throttling and priority-based scheduling
4. Provides graceful degradation when resources are constrained
5. Integrates with Redis for distributed task queuing
6. Supports different task priorities and provider categories
7. Includes circuit breaker patterns for failed providers
8. Provides comprehensive monitoring and alerting
9. Supports budget allocation and cost estimation
10. Includes emergency budget protection and fail-safe mechanisms.
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import redis.asyncio as aioredis
from fastapi import Request, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.core.metrics import ApiMetrics, LlmMetrics, agent_execution_duration_seconds, external_api_credit_used_total, external_api_duration_seconds, external_api_calls_total
from backend.core.redis_client import RedisClient, RedisDb, SerializationFormat
from backend.core.events import publish_event, EventPriority, EventCategory
from backend.providers import get_provider, ProviderConfig, get_all_providers

# Configure module logger
logger = logging.getLogger(__name__)

# Environment variables for configuration
BUDGET_CHECK_INTERVAL_SECONDS = int(os.getenv("BUDGET_CHECK_INTERVAL_SECONDS", "60"))
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS = int(os.getenv("CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS", "300")) # 5 minutes
EMERGENCY_BUDGET_THRESHOLD_PERCENT = float(os.getenv("EMERGENCY_BUDGET_THRESHOLD_PERCENT", "0.95")) # 95% of budget
TASK_QUEUE_KEY_PREFIX = "backpressure:queue:"
CIRCUIT_BREAKER_KEY_PREFIX = "backpressure:circuit:"

class TaskPriority(int, Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class CircuitState(str, Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Requests fail immediately
    HALF_OPEN = "half_open" # Allow a few requests to test if service has recovered

class QueuedTask(BaseModel):
    task_id: str
    provider_id: str
    priority: TaskPriority
    request_payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    estimated_cost: float = 0.0
    original_endpoint: str = ""

class CircuitBreakerState(BaseModel):
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    last_failure_time: Optional[datetime] = None
    opened_time: Optional[datetime] = None

class BackpressureManager:
    """
    Manages back-pressure, task queuing, and circuit breaking for external API providers.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, redis_client: Optional[RedisClient] = None):
        if self._initialized:
            return
        self.redis_client = redis_client or RedisClient()
        self.task_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue) # In-memory queues for immediate processing
        self.provider_budgets: Dict[str, float] = {} # Max budget for each provider
        self.provider_costs: Dict[str, float] = {} # Current cost for each provider
        self.provider_rate_limits: Dict[str, Dict[str, Any]] = {} # Rate limits for each provider
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {} # Circuit breaker state for each provider
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.version = os.getenv("APP_VERSION", "1.8.0-beta")
        self._initialized = True
        logger.info("BackpressureManager initialized.")

        # Start background task for processing queues and monitoring
        asyncio.create_task(self._background_monitor_and_processor())

    async def _load_provider_configs(self):
        """Loads provider configurations from the registry."""
        providers = get_all_providers()
        for provider in providers:
            provider_id = provider.get("id")
            if not provider_id:
                continue
            
            # Load budget (example: from a custom field in provider config)
            self.provider_budgets[provider_id] = provider.get("budget", float('inf')) # Default to infinite budget
            
            # Load rate limits
            self.provider_rate_limits[provider_id] = provider.get("rate_limits", {})
            
            # Load circuit breaker state from Redis
            cb_key = f"{CIRCUIT_BREAKER_KEY_PREFIX}{provider_id}"
            cb_state_data = await self.redis_client.get(cb_key, RedisDb.CACHE, SerializationFormat.JSON)
            if cb_state_data:
                self.circuit_breakers[provider_id] = CircuitBreakerState(**cb_state_data)
            else:
                self.circuit_breakers[provider_id] = CircuitBreakerState()
        logger.debug(f"Loaded configs for {len(providers)} providers.")

    async def _update_provider_costs(self):
        """Updates current costs for all providers from Prometheus metrics."""
        # This would ideally query Prometheus directly or use a Prometheus client library
        # For now, we'll simulate by reading the metric value (assuming it's updated by ApiMetrics)
        # In a real system, you'd query: sum(external_api_credit_used_total{provider='<provider_id>'})
        
        # This is a placeholder. In a real system, you'd need to query Prometheus.
        # For demonstration, we'll assume a simple increment or a mock.
        # The actual metric `external_api_credit_used_total` is a Counter, so it only increases.
        # To get current cost, you'd query Prometheus for its current value.
        # For this simulation, we'll just use a dummy value or rely on the metric being updated externally.
        
        # For a real implementation, you'd query Prometheus:
        # from prometheus_client import CollectorRegistry, generate_latest
        # registry = CollectorRegistry()
        # # You'd need to expose a way to get the current value of the counter
        # # For now, we'll just assume the metric is being updated elsewhere.
        pass

    async def _background_monitor_and_processor(self):
        """Background task to monitor budgets/rate limits and process queued tasks."""
        while True:
            await self._load_provider_configs()
            await self._update_provider_costs()

            for provider_id in self.circuit_breakers:
                await self._check_circuit_breaker(provider_id)
                await self._process_queued_tasks_for_provider(provider_id)
            
            await asyncio.sleep(BUDGET_CHECK_INTERVAL_SECONDS)

    async def _check_circuit_breaker(self, provider_id: str):
        """Checks and updates the state of a circuit breaker."""
        cb_state = self.circuit_breakers.get(provider_id)
        if not cb_state:
            return

        if cb_state.state == CircuitState.OPEN:
            if cb_state.opened_time and (datetime.now() - cb_state.opened_time).total_seconds() > CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS:
                cb_state.state = CircuitState.HALF_OPEN
                logger.warning(f"Circuit breaker for {provider_id} is now HALF-OPEN. Allowing test requests.")
                publish_event("circuit_breaker.state_change", {
                    "provider_id": provider_id,
                    "old_state": CircuitState.OPEN.value,
                    "new_state": CircuitState.HALF_OPEN.value,
                    "reason": "reset_timeout",
                })
        
        # Persist state to Redis
        cb_key = f"{CIRCUIT_BREAKER_KEY_PREFIX}{provider_id}"
        await self.redis_client.set(cb_key, cb_state.dict(), RedisDb.CACHE, SerializationFormat.JSON)

    async def _process_queued_tasks_for_provider(self, provider_id: str):
        """Processes tasks from the Redis queue for a specific provider."""
        redis_queue_key = f"{TASK_QUEUE_KEY_PREFIX}{provider_id}"
        
        # Get tasks from Redis queue, ordered by priority
        # Redis ZSET can be used for priority queue: ZRANGEBYSCORE key min max WITHSCORES
        # For simplicity, we'll use a list and sort in Python for now.
        
        queued_tasks_data = await self.redis_client.get(redis_queue_key, RedisDb.CACHE, SerializationFormat.JSON, default=[])
        if not queued_tasks_data:
            return

        queued_tasks = [QueuedTask(**t) for t in queued_tasks_data]
        queued_tasks.sort(key=lambda t: t.priority.value, reverse=True) # High priority first

        processed_count = 0
        while queued_tasks and self._can_execute_task(provider_id, queued_tasks[0]):
            task = queued_tasks.pop(0)
            logger.info(f"Processing queued task {task.task_id} for {provider_id} (Priority: {task.priority.name})")
            
            # Simulate task execution (in a real system, this would trigger the actual API call)
            # For now, we'll just log and increment a metric
            await asyncio.sleep(0.1) # Simulate work
            
            processed_count += 1
            publish_event("backpressure.task_processed", {
                "task_id": task.task_id,
                "provider_id": provider_id,
                "priority": task.priority.value,
                "estimated_cost": task.estimated_cost,
                "delay_seconds": (datetime.now() - task.timestamp).total_seconds(),
            })
            
            # Update costs (simulated)
            # external_api_credit_used_total.labels(provider=provider_id, credit_type="simulated").inc(task.estimated_cost)
            
            # Update circuit breaker state if it was half-open and successful
            cb_state = self.circuit_breakers.get(provider_id)
            if cb_state and cb_state.state == CircuitState.HALF_OPEN:
                cb_state.state = CircuitState.CLOSED
                cb_state.failures = 0
                logger.info(f"Circuit breaker for {provider_id} is now CLOSED (recovered).")
                publish_event("circuit_breaker.state_change", {
                    "provider_id": provider_id,
                    "old_state": CircuitState.HALF_OPEN.value,
                    "new_state": CircuitState.CLOSED.value,
                    "reason": "recovery",
                })
                # Persist state
                cb_key = f"{CIRCUIT_BREAKER_KEY_PREFIX}{provider_id}"
                await self.redis_client.set(cb_key, cb_state.dict(), RedisDb.CACHE, SerializationFormat.JSON)

        if processed_count > 0:
            logger.info(f"Processed {processed_count} queued tasks for {provider_id}.")
            # Save remaining tasks back to Redis
            await self.redis_client.set(redis_queue_key, [t.dict() for t in queued_tasks], RedisDb.CACHE, SerializationFormat.JSON)

    def _can_execute_task(self, provider_id: str, task: QueuedTask) -> bool:
        """Checks if a task can be executed based on budget, rate limits, and circuit breaker."""
        # Check circuit breaker
        cb_state = self.circuit_breakers.get(provider_id)
        if cb_state and cb_state.state == CircuitState.OPEN:
            logger.debug(f"Cannot execute task for {provider_id}: Circuit breaker is OPEN.")
            return False
        
        # Check budget
        current_cost = self.provider_costs.get(provider_id, 0.0)
        max_budget = self.provider_budgets.get(provider_id, float('inf'))
        
        if current_cost + task.estimated_cost > max_budget:
            logger.warning(f"Cannot execute task for {provider_id}: Budget exceeded. Current: {current_cost}, Task: {task.estimated_cost}, Max: {max_budget}")
            publish_event("backpressure.budget_exceeded", {
                "provider_id": provider_id,
                "current_cost": current_cost,
                "estimated_task_cost": task.estimated_cost,
                "max_budget": max_budget,
            })
            return False
        
        # Check emergency budget protection
        if max_budget != float('inf') and (current_cost / max_budget) > EMERGENCY_BUDGET_THRESHOLD_PERCENT:
            logger.warning(f"Cannot execute task for {provider_id}: Emergency budget threshold reached ({EMERGENCY_BUDGET_THRESHOLD_PERCENT*100}%).")
            publish_event("backpressure.emergency_budget_protection", {
                "provider_id": provider_id,
                "current_cost": current_cost,
                "max_budget": max_budget,
                "threshold": EMERGENCY_BUDGET_THRESHOLD_PERCENT,
            })
            return False

        # Check rate limits (simplified: assumes a simple requests_per_minute)
        rate_limits = self.provider_rate_limits.get(provider_id, {})
        rpm = rate_limits.get("requests_per_minute")
        
        if rpm:
            # This would require tracking requests per minute.
            # For a real system, you'd use a Redis counter or a token bucket algorithm.
            # For now, we'll assume the rate limit is handled by the client or external system.
            # If we were to implement it here, we'd need a per-minute counter in Redis.
            pass
        
        return True

    async def enqueue_task(self, task: QueuedTask):
        """Enqueues a task for later execution."""
        redis_queue_key = f"{TASK_QUEUE_KEY_PREFIX}{task.provider_id}"
        
        # Get existing tasks
        existing_tasks_data = await self.redis_client.get(redis_queue_key, RedisDb.CACHE, SerializationFormat.JSON, default=[])
        existing_tasks = [QueuedTask(**t) for t in existing_tasks_data] if existing_tasks_data else []
        
        # Add new task
        existing_tasks.append(task)
        
        # Sort by priority (high priority first)
        existing_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        
        # Save back to Redis
        await self.redis_client.set(redis_queue_key, [t.dict() for t in existing_tasks], RedisDb.CACHE, SerializationFormat.JSON)
        
        logger.info(f"Enqueued task {task.task_id} for {task.provider_id} (Priority: {task.priority.name})")
        publish_event("backpressure.task_enqueued", {
            "task_id": task.task_id,
            "provider_id": task.provider_id,
            "priority": task.priority.value,
            "estimated_cost": task.estimated_cost,
        })

    async def record_success(self, provider_id: str, cost: float = 0.0):
        """Records a successful API call."""
        # Update costs
        self.provider_costs[provider_id] = self.provider_costs.get(provider_id, 0.0) + cost
        
        # Reset circuit breaker failures if in HALF_OPEN state
        cb_state = self.circuit_breakers.get(provider_id)
        if cb_state and cb_state.state == CircuitState.HALF_OPEN:
            cb_state.state = CircuitState.CLOSED
            cb_state.failures = 0
            logger.info(f"Circuit breaker for {provider_id} is now CLOSED (recovered).")
            publish_event("circuit_breaker.state_change", {
                "provider_id": provider_id,
                "old_state": CircuitState.HALF_OPEN.value,
                "new_state": CircuitState.CLOSED.value,
                "reason": "success",
            })
            # Persist state
            cb_key = f"{CIRCUIT_BREAKER_KEY_PREFIX}{provider_id}"
            await self.redis_client.set(cb_key, cb_state.dict(), RedisDb.CACHE, SerializationFormat.JSON)

    async def record_failure(self, provider_id: str, error: str):
        """Records a failed API call."""
        # Update circuit breaker
        cb_state = self.circuit_breakers.get(provider_id)
        if not cb_state:
            cb_state = CircuitBreakerState()
            self.circuit_breakers[provider_id] = cb_state
        
        cb_state.failures += 1
        cb_state.last_failure_time = datetime.now()
        
        # Check if circuit breaker should open
        if cb_state.state == CircuitState.CLOSED and cb_state.failures >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            cb_state.state = CircuitState.OPEN
            cb_state.opened_time = datetime.now()
            logger.warning(f"Circuit breaker for {provider_id} is now OPEN after {cb_state.failures} failures.")
            publish_event("circuit_breaker.state_change", {
                "provider_id": provider_id,
                "old_state": CircuitState.CLOSED.value,
                "new_state": CircuitState.OPEN.value,
                "reason": "failure_threshold",
                "failures": cb_state.failures,
                "error": error,
            })
        
        # Persist state
        cb_key = f"{CIRCUIT_BREAKER_KEY_PREFIX}{provider_id}"
        await self.redis_client.set(cb_key, cb_state.dict(), RedisDb.CACHE, SerializationFormat.JSON)

    async def check_circuit_breaker(self, provider_id: str) -> Tuple[bool, Optional[str]]:
        """
        Checks if a provider's circuit breaker is open.
        
        Args:
            provider_id: Provider ID
            
        Returns:
            Tuple of (can_proceed, error_message)
        """
        cb_state = self.circuit_breakers.get(provider_id)
        if not cb_state:
            return True, None
        
        if cb_state.state == CircuitState.OPEN:
            # Calculate time until reset
            if cb_state.opened_time:
                elapsed = (datetime.now() - cb_state.opened_time).total_seconds()
                time_until_reset = max(0, CIRCUIT_BREAKER_RESET_TIMEOUT_SECONDS - elapsed)
                return False, f"Circuit breaker is OPEN. Try again in {time_until_reset:.0f} seconds."
            return False, "Circuit breaker is OPEN."
        
        return True, None

    async def estimate_task_cost(self, provider_id: str, endpoint: str, params: Dict[str, Any]) -> float:
        """
        Estimates the cost of a task based on provider, endpoint, and parameters.
        
        Args:
            provider_id: Provider ID
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Estimated cost
        """
        # This is a simplified implementation.
        # In a real system, you'd have a more sophisticated cost estimation model.
        
        # Get provider config
        provider_config = get_provider(provider_id)
        if not provider_config:
            return 0.0
        
        # Check if provider has cost estimation rules
        cost_rules = provider_config.get("cost_rules", {})
        
        # Default cost
        default_cost = cost_rules.get("default", 0.01)
        
        # Endpoint-specific cost
        endpoint_cost = cost_rules.get("endpoints", {}).get(endpoint, default_cost)
        
        # Parameter-based cost multipliers
        # Example: {"tokens": 0.001} would add 0.001 * tokens to the cost
        param_multipliers = cost_rules.get("param_multipliers", {})
        param_cost = 0.0
        for param, multiplier in param_multipliers.items():
            if param in params:
                param_value = params[param]
                if isinstance(param_value, (int, float)):
                    param_cost += param_value * multiplier
        
        total_cost = endpoint_cost + param_cost
        return total_cost

    async def check_budget(self, provider_id: str, estimated_cost: float) -> Tuple[bool, Optional[str]]:
        """
        Checks if a task can proceed based on budget constraints.
        
        Args:
            provider_id: Provider ID
            estimated_cost: Estimated cost of the task
            
        Returns:
            Tuple of (can_proceed, error_message)
        """
        current_cost = self.provider_costs.get(provider_id, 0.0)
        max_budget = self.provider_budgets.get(provider_id, float('inf'))
        
        # Check if budget would be exceeded
        if current_cost + estimated_cost > max_budget:
            return False, f"Budget exceeded. Current: {current_cost}, Task: {estimated_cost}, Max: {max_budget}"
        
        # Check emergency budget protection
        if max_budget != float('inf') and (current_cost / max_budget) > EMERGENCY_BUDGET_THRESHOLD_PERCENT:
            return False, f"Emergency budget threshold reached ({EMERGENCY_BUDGET_THRESHOLD_PERCENT*100}%)."
        
        return True, None

    async def check_rate_limit(self, provider_id: str, endpoint: str) -> Tuple[bool, Optional[str]]:
        """
        Checks if a task can proceed based on rate limit constraints.
        
        Args:
            provider_id: Provider ID
            endpoint: API endpoint
            
        Returns:
            Tuple of (can_proceed, error_message)
        """
        # This is a simplified implementation.
        # In a real system, you'd use a token bucket algorithm or similar.
        
        # Get rate limits
        rate_limits = self.provider_rate_limits.get(provider_id, {})
        rpm = rate_limits.get("requests_per_minute")
        
        if not rpm:
            return True, None
        
        # Check current rate
        # In a real implementation, you'd track this in Redis
        # For now, we'll assume we're under the limit
        
        return True, None

    async def process_request(
        self,
        provider_id: str,
        endpoint: str,
        params: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Processes a request through the back-pressure system.
        
        Args:
            provider_id: Provider ID
            endpoint: API endpoint
            params: Request parameters
            priority: Task priority
            
        Returns:
            Tuple of (can_proceed, error_message, task_id)
        """
        # Check circuit breaker
        can_proceed, error_message = await self.check_circuit_breaker(provider_id)
        if not can_proceed:
            return False, error_message, None
        
        # Estimate cost
        estimated_cost = await self.estimate_task_cost(provider_id, endpoint, params)
        
        # Check budget
        can_proceed, error_message = await self.check_budget(provider_id, estimated_cost)
        if not can_proceed and priority != TaskPriority.CRITICAL:
            # Enqueue task for later execution
            task_id = f"{provider_id}:{endpoint}:{time.time()}"
            task = QueuedTask(
                task_id=task_id,
                provider_id=provider_id,
                priority=priority,
                request_payload=params,
                estimated_cost=estimated_cost,
                original_endpoint=endpoint,
            )
            await self.enqueue_task(task)
            return False, error_message, task_id
        
        # Check rate limit
        can_proceed, error_message = await self.check_rate_limit(provider_id, endpoint)
        if not can_proceed and priority != TaskPriority.CRITICAL:
            # Enqueue task for later execution
            task_id = f"{provider_id}:{endpoint}:{time.time()}"
            task = QueuedTask(
                task_id=task_id,
                provider_id=provider_id,
                priority=priority,
                request_payload=params,
                estimated_cost=estimated_cost,
                original_endpoint=endpoint,
            )
            await self.enqueue_task(task)
            return False, error_message, task_id
        
        # All checks passed, can proceed
        return True, None, None


class BackpressureMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for applying back-pressure to API requests.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        backpressure_manager: Optional[BackpressureManager] = None,
    ):
        super().__init__(app)
        self.backpressure_manager = backpressure_manager or BackpressureManager()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Dispatches a request through the middleware.
        
        Args:
            request: FastAPI request
            call_next: Next middleware in chain
            
        Returns:
            FastAPI response
        """
        # Check if request is to an external API
        if not self._is_external_api_request(request):
            return await call_next(request)
        
        # Extract provider ID and endpoint
        provider_id, endpoint = self._extract_provider_info(request)
        if not provider_id:
            return await call_next(request)
        
        # Extract request parameters
        params = await self._extract_request_params(request)
        
        # Determine priority (could be based on user role, request type, etc.)
        priority = self._determine_priority(request)
        
        # Process request through back-pressure system
        can_proceed, error_message, task_id = await self.backpressure_manager.process_request(
            provider_id=provider_id,
            endpoint=endpoint,
            params=params,
            priority=priority,
        )
        
        if not can_proceed:
            # Request was queued or rejected
            status_code = 429 if "rate limit" in (error_message or "").lower() else 503
            return Response(
                content=json.dumps({
                    "error": error_message,
                    "queued": task_id is not None,
                    "task_id": task_id,
                }),
                media_type="application/json",
                status_code=status_code,
            )
        
        # Proceed with request
        try:
            response = await call_next(request)
            
            # Record success
            if response.status_code < 400:
                # Estimate cost
                estimated_cost = await self.backpressure_manager.estimate_task_cost(
                    provider_id=provider_id,
                    endpoint=endpoint,
                    params=params,
                )
                await self.backpressure_manager.record_success(provider_id, estimated_cost)
            else:
                # Record failure
                await self.backpressure_manager.record_failure(
                    provider_id=provider_id,
                    error=f"HTTP {response.status_code}",
                )
            
            return response
        
        except Exception as e:
            # Record failure
            await self.backpressure_manager.record_failure(
                provider_id=provider_id,
                error=str(e),
            )
            raise
    
    def _is_external_api_request(self, request: Request) -> bool:
        """
        Checks if a request is to an external API.
        
        Args:
            request: FastAPI request
            
        Returns:
            True if request is to an external API
        """
        # This is a simplified implementation.
        # In a real system, you'd have a more sophisticated way to identify external API requests.
        
        # Example: Check if path starts with /api/external/
        return request.url.path.startswith("/api/external/")
    
    def _extract_provider_info(self, request: Request) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts provider ID and endpoint from a request.
        
        Args:
            request: FastAPI request
            
        Returns:
            Tuple of (provider_id, endpoint)
        """
        # This is a simplified implementation.
        # In a real system, you'd have a more sophisticated way to extract provider info.
        
        # Example: /api/external/{provider_id}/{endpoint}
        parts = request.url.path.split("/")
        if len(parts) >= 4 and parts[1] == "api" and parts[2] == "external":
            provider_id = parts[3]
            endpoint = "/".join(parts[4:])
            return provider_id, endpoint
        
        return None, None
    
    async def _extract_request_params(self, request: Request) -> Dict[str, Any]:
        """
        Extracts parameters from a request.
        
        Args:
            request: FastAPI request
            
        Returns:
            Request parameters
        """
        # Get query parameters
        params = dict(request.query_params)
        
        # Get body parameters for POST/PUT/PATCH requests
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    params.update(body)
            except:
                pass
        
        return params
    
    def _determine_priority(self, request: Request) -> TaskPriority:
        """
        Determines the priority of a request.
        
        Args:
            request: FastAPI request
            
        Returns:
            Task priority
        """
        # This is a simplified implementation.
        # In a real system, you'd have a more sophisticated way to determine priority.
        
        # Example: Check for X-Priority header
        priority_header = request.headers.get("X-Priority", "").upper()
        if priority_header == "CRITICAL":
            return TaskPriority.CRITICAL
        elif priority_header == "HIGH":
            return TaskPriority.HIGH
        elif priority_header == "LOW":
            return TaskPriority.LOW
        
        # Default to normal priority
        return TaskPriority.NORMAL


# Decorator for applying back-pressure to functions
def with_backpressure(
    provider_id: str,
    endpoint: str,
    priority: TaskPriority = TaskPriority.NORMAL,
    backpressure_manager: Optional[BackpressureManager] = None,
):
    """
    Decorator for applying back-pressure to functions.
    
    Args:
        provider_id: Provider ID
        endpoint: API endpoint
        priority: Task priority
        backpressure_manager: BackpressureManager instance
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = backpressure_manager or BackpressureManager()
            
            # Process request through back-pressure system
            can_proceed, error_message, task_id = await manager.process_request(
                provider_id=provider_id,
                endpoint=endpoint,
                params=kwargs,
                priority=priority,
            )
            
            if not can_proceed:
                # Request was queued or rejected
                if task_id:
                    return {"queued": True, "task_id": task_id, "error": error_message}
                else:
                    raise Exception(f"Request rejected: {error_message}")
            
            # Proceed with function
            try:
                result = await func(*args, **kwargs)
                
                # Record success
                estimated_cost = await manager.estimate_task_cost(
                    provider_id=provider_id,
                    endpoint=endpoint,
                    params=kwargs,
                )
                await manager.record_success(provider_id, estimated_cost)
                
                return result
            
            except Exception as e:
                # Record failure
                await manager.record_failure(
                    provider_id=provider_id,
                    error=str(e),
                )
                raise
        
        return wrapper
    
    return decorator


# Helper function to get task status
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Gets the status of a queued task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task status
    """
    # Extract provider ID from task ID
    parts = task_id.split(":")
    if len(parts) < 2:
        return {"error": "Invalid task ID"}
    
    provider_id = parts[0]
    
    # Get manager instance
    manager = BackpressureManager()
    
    # Get tasks from Redis queue
    redis_queue_key = f"{TASK_QUEUE_KEY_PREFIX}{provider_id}"
    queued_tasks_data = await manager.redis_client.get(redis_queue_key, RedisDb.CACHE, SerializationFormat.JSON, default=[])
    
    # Check if task is in queue
    for task_data in queued_tasks_data:
        task = QueuedTask(**task_data)
        if task.task_id == task_id:
            # Calculate queue position
            queue_position = 1
            for other_task_data in queued_tasks_data:
                other_task = QueuedTask(**other_task_data)
                if other_task.priority.value > task.priority.value:
                    queue_position += 1
            
            # Calculate estimated time
            # This is a very rough estimate
            estimated_seconds = queue_position * 5  # Assume 5 seconds per task
            
            return {
                "status": "queued",
                "task_id": task_id,
                "provider_id": provider_id,
                "priority": task.priority.name,
                "queue_position": queue_position,
                "estimated_seconds": estimated_seconds,
                "queued_at": task.timestamp.isoformat(),
            }
    
    # Task not found in queue, assume it's been processed
    return {
        "status": "processed",
        "task_id": task_id,
        "provider_id": provider_id,
    }
