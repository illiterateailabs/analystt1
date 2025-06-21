"""
Crew API endpoints for managing CrewAI crews.

This module provides endpoints for running, pausing, and resuming CrewAI crews,
as well as retrieving crew results and status. It integrates with the
Graph-Aware RAG, Evidence, HITL, OpenTelemetry, and Backpressure systems.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Request, status, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from backend.agents.custom_crew import CustomCrew, CrewMode
from backend.agents.factory import CrewFactory, RUNNING_CREWS
from backend.auth.rbac import require_roles, Roles, RoleSets
from backend.core import telemetry
from backend.core.backpressure import BackpressureManager, QueuedTask, TaskPriority
from backend.core.events import publish_event, EventPriority, EventCategory
from backend.core.metrics import ApiMetrics, AgentMetrics
from backend.core.graph_rag import GraphRAG
from backend.core.evidence import EvidenceBundle, create_evidence_bundle
from backend.core.redis_client import RedisClient, RedisDb, SerializationFormat
from backend.api.v1.hitl import create_hitl_callback, HITLReviewRepository, ReviewStatus, ReviewResponse
from backend.database import get_db
from sqlalchemy.orm import Session

# Configure logging
logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize core services (these should ideally be injected via FastAPI dependencies)
# For simplicity and to avoid circular imports, we'll initialize them here.
# In a larger application, a dependency injection framework would manage these.
_graph_rag_service: Optional[GraphRAG] = None
_redis_client: Optional[RedisClient] = None
_backpressure_manager: Optional[BackpressureManager] = None

def get_graph_rag_service() -> GraphRAG:
    global _graph_rag_service
    if _graph_rag_service is None:
        _graph_rag_service = GraphRAG()
    return _graph_rag_service

def get_redis_client() -> RedisClient:
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client

def get_backpressure_manager() -> BackpressureManager:
    global _backpressure_manager
    if _backpressure_manager is None:
        _backpressure_manager = BackpressureManager(redis_client=get_redis_client())
    return _backpressure_manager


# Request/Response Models
class CrewRunRequest(BaseModel):
    """Request model for running a crew."""
    crew_name: str = Field(..., description="Name of the crew to run")
    workflow_name: Optional[str] = Field(None, description="Specific workflow to run within the crew")
    inputs: Optional[Dict[str, Any]] = Field(None, description="Inputs for the crew")
    mode: Optional[CrewMode] = Field(None, description="Execution mode for the crew (sequential, hierarchical, planning)")
    priority: TaskPriority = Field(TaskPriority.NORMAL, description="Priority of the crew execution")


class CrewRunResponse(BaseModel):
    """Response model for a successful crew run or a queued task."""
    task_id: str = Field(..., description="ID of the crew task")
    status: str = Field(..., description="Status of the operation (running, queued, error)")
    message: str = Field(..., description="Descriptive message about the operation")
    result: Optional[Dict[str, Any]] = Field(None, description="Final result of the crew execution (if completed synchronously)")
    evidence_bundle_id: Optional[str] = Field(None, description="ID of the generated evidence bundle")
    error: Optional[str] = Field(None, description="Error message if any")


class CrewPauseRequest(BaseModel):
    """Request model for pausing a crew."""
    task_id: str = Field(..., description="ID of the task to pause")
    reason: Optional[str] = Field(None, description="Reason for pausing")
    review_id: Optional[str] = Field(None, description="ID of the associated review")


class CrewResumeRequest(BaseModel):
    """Request model for resuming a crew."""
    task_id: str = Field(..., description="ID of the task to resume")
    review_result: Optional[Dict[str, Any]] = Field(None, description="Result of the review")


class CrewResponse(BaseModel):
    """Generic response model for crew operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    task_id: Optional[str] = Field(None, description="ID of the task")
    result: Optional[str] = Field(None, description="Result of the crew execution")
    error: Optional[str] = Field(None, description="Error message if any")


class TaskListItem(BaseModel):
    """Model for a task in the task list."""
    task_id: str = Field(..., description="Task ID")
    crew_name: str = Field(..., description="Name of the crew")
    state: str = Field(..., description="Current state of the task")
    start_time: str = Field(..., description="Time when the task started")
    last_updated: str = Field(..., description="Time when the task was last updated")
    current_agent: Optional[str] = Field(None, description="Current agent processing the task")
    error: Optional[str] = Field(None, description="Error message if any")
    review_id: Optional[str] = Field(None, description="ID of associated review if paused for HITL")


class TaskListResponse(BaseModel):
    """Response model for listing tasks."""
    tasks: List[TaskListItem] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks")


class TaskResultResponse(BaseModel):
    """Response model for task results."""
    task_id: str = Field(..., description="Task ID")
    crew_name: str = Field(..., description="Name of the crew")
    state: str = Field(..., description="Current state of the task")
    start_time: str = Field(..., description="Time when the task started")
    completion_time: Optional[str] = Field(None, description="Time when the task completed")
    result: Optional[str] = Field(None, description="Raw result of the crew execution")
    report: Optional[str] = Field(None, description="Formatted report from the crew")
    visualizations: Optional[List[Dict[str, Any]]] = Field(None, description="Visualizations generated by the crew")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the task")


# Endpoints
@router.post(
    "/execute",
    response_model=CrewRunResponse,
    summary="Execute a crew",
    description="Execute a CrewAI crew with the specified inputs, integrating with RAG, Evidence, and HITL."
)
async def execute_crew(
    request: CrewRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    backpressure_manager: BackpressureManager = Depends(get_backpressure_manager),
    graph_rag_service: GraphRAG = Depends(get_graph_rag_service),
    redis_client: RedisClient = Depends(get_redis_client),
    roles: RoleSets = Depends(require_roles([Roles.ANALYST, Roles.ADMIN]))
) -> CrewRunResponse:
    """
    Execute a CrewAI crew.

    Args:
        request (CrewRunRequest): Request model with crew name, workflow, inputs, and mode.
        background_tasks: FastAPI BackgroundTasks for async operations.
        db: Database session for HITL integration.
        backpressure_manager: Instance of BackpressureManager.
        graph_rag_service: Instance of GraphRAG.
        redis_client: Instance of RedisClient.

    Returns:
        CrewRunResponse: Response with task ID, status, and result.
    """
    span_name = f"crew.execute.{request.crew_name}"
    async with telemetry.trace_async_operation(span_name, attributes={"crew.name": request.crew_name, "crew.workflow": request.workflow_name, "crew.mode": request.mode.value if request.mode else "default"}) as span:
        ApiMetrics.track_call(
            provider="internal",
            endpoint="/api/v1/crew/execute",
            func=lambda: None,
            environment=telemetry.os.environ.get("ENVIRONMENT", "development"),
            version=telemetry.os.environ.get("APP_VERSION", "1.8.0-beta"),
        )()

        # Create a unique task ID for this execution
        task_id = f"crew_{request.crew_name}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        span.set_attribute("crew.task_id", task_id)

        # Prepare a dummy task for backpressure estimation (actual cost will be tracked by ApiMetrics)
        dummy_task = QueuedTask(
            task_id=task_id,
            provider_id=request.crew_name, # Use crew name as provider for backpressure
            priority=request.priority,
            request_payload=request.inputs or {},
            estimated_cost=0.01 # Small estimated cost for initial check
        )

        # Check backpressure before starting the crew
        can_proceed, error_message, queued_task_id = await backpressure_manager.process_request(
            provider_id=request.crew_name,
            endpoint="crew_execution",
            params=request.inputs or {},
            priority=request.priority
        )

        if not can_proceed:
            if queued_task_id:
                span.set_attribute("crew.status", "queued")
                return CrewRunResponse(
                    task_id=queued_task_id,
                    status="queued",
                    message=f"Crew execution queued due to backpressure: {error_message}",
                    error=error_message
                )
            else:
                span.set_attribute("crew.status", "rejected")
                span.set_status(telemetry.trace.Status(telemetry.trace.StatusCode.ERROR, error_message))
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Crew execution rejected: {error_message}"
                )

        # Initialize CrewFactory and run the crew in a background task
        async def _run_crew_in_background():
            try:
                # Initialize services for the background task
                local_graph_rag = get_graph_rag_service()
                local_redis_client = get_redis_client()
                local_backpressure_manager = get_backpressure_manager()
                local_hitl_callback = create_hitl_callback(db)

                # Create a new evidence bundle for this crew run
                evidence_bundle = create_evidence_bundle(
                    title=f"Crew Execution: {request.crew_name}",
                    description=f"Evidence bundle for {request.crew_name} execution with inputs: {request.inputs}",
                    source_type="crew_execution",
                    source_id=task_id
                )

                # Initialize custom crew with all our services
                crew = CustomCrew(
                    crew_name=request.crew_name,
                    task_id=task_id,
                    mode=request.mode or CrewMode.SEQUENTIAL,
                    graph_rag=local_graph_rag,
                    redis_client=local_redis_client,
                    hitl_callback=local_hitl_callback,
                    evidence_bundle=evidence_bundle
                )

                # Publish event for crew execution start
                publish_event(
                    event_type="crew.execution.start",
                    payload={
                        "task_id": task_id,
                        "crew_name": request.crew_name,
                        "workflow": request.workflow_name,
                        "inputs": request.inputs,
                        "timestamp": datetime.now().isoformat()
                    },
                    priority=EventPriority.HIGH,
                    category=EventCategory.CREW
                )

                # Run the crew
                result = await crew.run(
                    workflow_name=request.workflow_name,
                    inputs=request.inputs or {}
                )

                # Update evidence bundle with results
                evidence_bundle.add_narrative(f"Crew execution completed with result: {result}")
                evidence_bundle_id = evidence_bundle.id

                # Save evidence bundle to Redis for persistence
                redis_client.set(
                    key=f"evidence:bundle:{evidence_bundle_id}",
                    value=evidence_bundle.dict(),
                    db=RedisDb.CACHE,
                    format=SerializationFormat.JSON
                )

                # Update task data with evidence bundle ID
                if task_id in RUNNING_CREWS:
                    RUNNING_CREWS[task_id]["evidence_bundle_id"] = evidence_bundle_id
                    RUNNING_CREWS[task_id]["result"] = result
                    RUNNING_CREWS[task_id]["state"] = "COMPLETED"
                    RUNNING_CREWS[task_id]["completion_time"] = datetime.now().isoformat()

                # Record success with backpressure manager
                await local_backpressure_manager.record_success(
                    provider_id=request.crew_name,
                    cost=0.0  # Actual cost tracking would be done via metrics
                )

                # Publish event for crew execution completion
                publish_event(
                    event_type="crew.execution.complete",
                    payload={
                        "task_id": task_id,
                        "crew_name": request.crew_name,
                        "workflow": request.workflow_name,
                        "result": result,
                        "evidence_bundle_id": evidence_bundle_id,
                        "timestamp": datetime.now().isoformat()
                    },
                    priority=EventPriority.HIGH,
                    category=EventCategory.CREW
                )

                # Track execution metrics
                AgentMetrics.track_execution(
                    agent_type=f"crew.{request.crew_name}",
                    operation="execute",
                    success=True,
                    duration_ms=(datetime.now() - datetime.fromisoformat(RUNNING_CREWS[task_id]["start_time"])).total_seconds() * 1000
                )

                logger.info(f"Crew execution completed: {task_id}")

            except Exception as e:
                logger.error(f"Crew execution failed: {e}", exc_info=True)

                # Update task data with error
                if task_id in RUNNING_CREWS:
                    RUNNING_CREWS[task_id]["state"] = "ERROR"
                    RUNNING_CREWS[task_id]["error"] = str(e)
                    RUNNING_CREWS[task_id]["completion_time"] = datetime.now().isoformat()

                # Record failure with backpressure manager
                await local_backpressure_manager.record_failure(
                    provider_id=request.crew_name,
                    error=str(e)
                )

                # Publish event for crew execution error
                publish_event(
                    event_type="crew.execution.error",
                    payload={
                        "task_id": task_id,
                        "crew_name": request.crew_name,
                        "workflow": request.workflow_name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    },
                    priority=EventPriority.HIGH,
                    category=EventCategory.CREW
                )

                # Track execution metrics
                AgentMetrics.track_execution(
                    agent_type=f"crew.{request.crew_name}",
                    operation="execute",
                    success=False,
                    duration_ms=(datetime.now() - datetime.fromisoformat(RUNNING_CREWS[task_id]["start_time"])).total_seconds() * 1000
                )

        # Initialize task data
        RUNNING_CREWS[task_id] = {
            "crew_name": request.crew_name,
            "state": "RUNNING",
            "start_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "inputs": request.inputs,
            "workflow": request.workflow_name,
            "context": {},
        }

        # Start background task
        background_tasks.add_task(_run_crew_in_background)

        # Return response
        span.set_attribute("crew.status", "running")
        return CrewRunResponse(
            task_id=task_id,
            status="running",
            message=f"Crew execution started: {request.crew_name}",
            result=None,
            evidence_bundle_id=None
        )


@router.post(
    "/pause",
    response_model=CrewResponse,
    summary="Pause a crew",
    description="Pause a running CrewAI crew."
)
async def pause_crew(
    request: CrewPauseRequest,
    db: Session = Depends(get_db),
    backpressure_manager: BackpressureManager = Depends(get_backpressure_manager),
    roles: RoleSets = Depends(require_roles([Roles.ANALYST, Roles.ADMIN]))
):
    """
    Pause a running CrewAI crew.
    
    Args:
        request (CrewPauseRequest): Request model with task ID and reason.
        
    Returns:
        CrewResponse: Response indicating success or failure.
    """
    async with telemetry.trace_async_operation("crew.pause", attributes={"task_id": request.task_id}) as span:
        ApiMetrics.track_call(
            provider="internal",
            endpoint="/api/v1/crew/pause",
            func=lambda: None,
            environment=telemetry.os.environ.get("ENVIRONMENT", "development"),
            version=telemetry.os.environ.get("APP_VERSION", "1.8.0-beta"),
        )()

        try:
            # Check if task exists
            if request.task_id not in RUNNING_CREWS:
                span.set_status(telemetry.trace.Status(telemetry.trace.StatusCode.ERROR, "Task not found"))
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Task '{request.task_id}' not found"
                )
            
            # Get task data
            task_data = RUNNING_CREWS[request.task_id]
            
            # Check if task can be paused
            if task_data.get("state") not in ["RUNNING", "WAITING"]:
                span.set_status(telemetry.trace.Status(telemetry.trace.StatusCode.ERROR, f"Cannot pause task in state {task_data.get('state')}"))
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot pause task in state {task_data.get('state')}"
                )
            
            # Pause crew
            success = CrewFactory.pause_crew(request.task_id, request.reason, request.review_id)
            
            if success:
                # Update task data
                task_data["state"] = "PAUSED"
                task_data["paused_at"] = datetime.now().isoformat()
                task_data["last_updated"] = datetime.now().isoformat()
                task_data["pause_reason"] = request.reason
                task_data["review_id"] = request.review_id
                
                # If review ID is provided, update HITL review
                if request.review_id:
                    hitl_repo = HITLReviewRepository(db)
                    await hitl_repo.update_review_status(
                        review_id=request.review_id,
                        status=ReviewStatus.PENDING,
                        reviewer_id=None
                    )
                
                # Publish event
                publish_event(
                    event_type="crew.execution.paused",
                    payload={
                        "task_id": request.task_id,
                        "crew_name": task_data.get("crew_name"),
                        "reason": request.reason,
                        "review_id": request.review_id,
                        "timestamp": datetime.now().isoformat()
                    },
                    priority=EventPriority.MEDIUM,
                    category=EventCategory.CREW
                )
                
                span.set_attribute("crew.paused", True)
                span.set_attribute("crew.review_id", request.review_id)
            else:
                span.set_status(telemetry.trace.Status(telemetry.trace.StatusCode.ERROR, "Failed to pause crew"))
            
            # Create response
            response = CrewResponse(
                success=success,
                task_id=request.task_id,
                error=None if success else "Failed to pause crew"
            )
            
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to pause crew: {e}", exc_info=True)
            span.set_status(telemetry.trace.Status(telemetry.trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to pause crew: {str(e)}"
            )


@router.post(
    "/resume",
    response_model=CrewResponse,
    summary="Resume a crew",
    description="Resume a paused CrewAI crew."
)
async def resume_crew(
    request: CrewResumeRequest,
    db: Session = Depends(get_db),
    backpressure_manager: BackpressureManager = Depends(get_backpressure_manager),
    roles: RoleSets = Depends(require_roles([Roles.ANALYST, Roles.ADMIN]))
):
    """
    Resume a paused CrewAI crew.
    
    Args:
        request (CrewResumeRequest): Request model with task ID and review result.
        
    Returns:
        CrewResponse: Response indicating success or failure.
    """
    async with telemetry.trace_async_operation("crew.resume", attributes={"task_id": request.task_id}) as span:
        ApiMetrics.track_call(
            provider="internal",
            endpoint="/api/v1/crew/resume",
            func=lambda: None,
            environment=telemetry.os.environ.get("ENVIRONMENT", "development"),
            version=telemetry.os.environ.get("APP_VERSION", "1.8.0-beta"),
        )()

        try:
            # Check if task exists
            if request.task_id not in RUNNING_CREWS:
                span.set_status(telemetry.trace.Status(telemetry.trace.StatusCode.ERROR, "Task not found"))
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Task '{request.task_id}' not found"
                )
            
            # Get task data
            task_data = RUNNING_CREWS[request.task_id]
            
            # Check if task can be resumed
            if task_data.get("state") != "PAUSED":
                span.set_status(telemetry.trace.Status(telemetry.trace.StatusCode.ERROR, f"Cannot resume task in state {task_data.get('state')}"))
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot resume task in state {task_data.get('state')}"
                )
            
            # Resume crew
            success = CrewFactory.resume_crew(request.task_id, request.review_result)
            
            if success:
                # Update task data
                task_data["state"] = "RUNNING"
                task_data["resumed_at"] = datetime.now().isoformat()
                task_data["last_updated"] = datetime.now().isoformat()
                
                # If review ID is provided, update HITL review
                if task_data.get("review_id"):
                    hitl_repo = HITLReviewRepository(db)
                    await hitl_repo.update_review_status(
                        review_id=task_data["review_id"],
                        status=ReviewStatus.APPROVED,
                        reviewer_id=None
                    )
                    
                    # Add review result to task data
                    if request.review_result:
                        if "review_results" not in task_data:
                            task_data["review_results"] = []
                        task_data["review_results"].append({
                            "review_id": task_data["review_id"],
                            "result": request.review_result,
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Publish event
                publish_event(
                    event_type="crew.execution.resumed",
                    payload={
                        "task_id": request.task_id,
                        "crew_name": task_data.get("crew_name"),
                        "review_result": request.review_result,
                        "timestamp": datetime.now().isoformat()
                    },
                    priority=EventPriority.MEDIUM,
                    category=EventCategory.CREW
                )
                
                span.set_attribute("crew.resumed", True)
            else:
                span.set_status(telemetry.trace.Status(telemetry.trace.StatusCode.ERROR, "Failed to resume crew"))
            
            # Create response
            response = CrewResponse(
                success=success,
                task_id=request.task_id,
                error=None if success else "Failed to resume crew"
            )
            
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to resume crew: {e}", exc_info=True)
            span.set_status(telemetry.trace.Status(telemetry.trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to resume crew: {str(e)}"
            )


@router.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="List tasks",
    description="List all crew tasks with their current status."
)
async def list_tasks(
    state: Optional[str] = None,
    crew_name: Optional[str] = None,
    roles: RoleSets = Depends(require_roles([Roles.ANALYST, Roles.ADMIN]))
):
    """
    List all crew tasks with their current status.
    
    Args:
        state (str, optional): Filter by task state. Defaults to None.
        crew_name (str, optional): Filter by crew name. Defaults to None.
        
    Returns:
        TaskListResponse: List of tasks.
    """
    async with telemetry.trace_async_operation("crew.list_tasks", attributes={"state": state, "crew_name": crew_name}) as span:
        ApiMetrics.track_call(
            provider="internal",
            endpoint="/api/v1/crew/tasks",
            func=lambda: None,
            environment=telemetry.os.environ.get("ENVIRONMENT", "development"),
            version=telemetry.os.environ.get("APP_VERSION", "1.8.0-beta"),
        )()

        try:
            # Filter tasks
            filtered_tasks = []
            for task_id, task_data in RUNNING_CREWS.items():
                # Apply filters
                if state and task_data.get("state") != state:
                    continue
                if crew_name and task_data.get("crew_name") != crew_name:
                    continue
                
                # Add to list
                filtered_tasks.append(TaskListItem(
                    task_id=task_id,
                    crew_name=task_data.get("crew_name", "unknown"),
                    state=task_data.get("state", "UNKNOWN"),
                    start_time=task_data.get("start_time", datetime.now().isoformat()),
                    last_updated=task_data.get("last_updated", datetime.now().isoformat()),
                    current_agent=task_data.get("current_agent"),
                    error=task_data.get("error"),
                    review_id=task_data.get("review_id")
                ))
            
            # Sort by start_time (newest first)
            filtered_tasks.sort(key=lambda x: x.start_time, reverse=True)
            
            span.set_attribute("tasks.count", len(filtered_tasks))
            
            return TaskListResponse(
                tasks=filtered_tasks,
                total=len(filtered_tasks)
            )
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}", exc_info=True)
            span.set_status(telemetry.trace.Status(telemetry.trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list tasks: {str(e)}"
            )


@router.get(
    "/{task_id}/result",
    response_model=TaskResultResponse,
    summary="Get task result",
    description="Get the result of a crew task."
)
async def get_task_result(
    task_id: str,
    roles: RoleSets = Depends(require_roles([Roles.ANALYST, Roles.ADMIN]))
):
    """
    Get the result of a crew task.
    
    Args:
        task_id (str): ID of the task.
        
    Returns:
        TaskResultResponse: Task result.
    """
    async with telemetry.trace_async_operation("crew.get_task_result", attributes={"task_id": task_id}) as span:
        ApiMetrics.track_call(
            provider="internal",
            endpoint="/api/v1/crew/{task_id}/result",
            func=lambda: None,
            environment=telemetry.os.environ.get("ENVIRONMENT", "development"),
            version=telemetry.os.environ.get("APP_VERSION", "1.8.0-beta"),
        )()

        try:
            # Check if task exists
            if task_id not in RUNNING_CREWS:
                span.set_status(telemetry.trace.Status(telemetry.trace.StatusCode.ERROR, "Task not found"))
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Task '{task_id}' not found"
                )
            
            # Get task data
            task_data = RUNNING_CREWS[task_id]
            
            # Extract result
            result = task_data.get("result", "")
            
            # Extract report (from context if available)
            report = ""
            if "_context" in task_data.get("context", {}) and "report" in task_data["context"]["_context"]:
                report = task_data["context"]["_context"]["report"]
            
            # Extract visualizations (from context if available)
            visualizations = []
            if "_context" in task_data.get("context", {}) and "visualizations" in task_data["context"]["_context"]:
                visualizations = task_data["context"]["_context"]["visualizations"]
            
            # Build metadata
            metadata = {}
            if "inputs" in task_data:
                metadata["inputs"] = task_data["inputs"]
            
            # Add evidence bundle ID if available
            if "evidence_bundle_id" in task_data:
                metadata["evidence_bundle_id"] = task_data["evidence_bundle_id"]
            
            # Add pause duration if applicable
            if task_data.get("paused_at") and task_data.get("resumed_at"):
                try:
                    paused_at = datetime.fromisoformat(task_data["paused_at"])
                    resumed_at = datetime.fromisoformat(task_data["resumed_at"])
                    metadata["paused_duration"] = (resumed_at - paused_at).total_seconds()
                except Exception:
                    pass
            
            span.set_attribute("task.state", task_data.get("state", "UNKNOWN"))
            if "evidence_bundle_id" in task_data:
                span.set_attribute("task.evidence_bundle_id", task_data["evidence_bundle_id"])
            
            # Create response
            response = TaskResultResponse(
                task_id=task_id,
                crew_name=task_data.get("crew_name", "unknown"),
                state=task_data.get("state", "UNKNOWN"),
                start_time=task_data.get("start_time", datetime.now().isoformat()),
                completion_time=task_data.get("completion_time"),
                result=result,
                report=report,
                visualizations=visualizations if visualizations else None,
                metadata=metadata
            )
            
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get task result: {e}", exc_info=True)
            span.set_status(telemetry.trace.Status(telemetry.trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get task result: {str(e)}"
            )
