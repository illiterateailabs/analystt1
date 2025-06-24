"""
API Endpoints for Anomaly Detection and Fraud Monitoring

This module provides FastAPI endpoints for interacting with the Anomaly Detection Service.
It includes functionalities for:
- Triggering on-demand anomaly detection for single or bulk entities
- Managing anomaly alerts (create, retrieve, update, list)
- Managing detection strategies (create, retrieve, update, list, delete)
- Real-time streaming of anomaly alerts via WebSocket
- Providing dashboard-ready anomaly statistics
- Managing background anomaly detection jobs
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.auth.dependencies import get_current_user
from backend.core.anomaly_detection import (
    AnomalyDetectionService,
    AnomalyDetectionResult,
    AnomalyAlert,
    AnomalySeverity,
    DetectionStrategy,
    DataEntityType,
    AnomalyType,
    DetectionMethod,
    AlertStatus,
)
from backend.jobs.tasks.anomaly_tasks import (
    detect_single_entity_anomaly_task,
    detect_bulk_anomalies_task,
    continuous_anomaly_monitoring_task,
    generate_anomaly_alert_task,
    update_alert_status_task,
    add_detection_strategy_task,
    update_detection_strategy_task,
    delete_detection_strategy_task,
    notify_anomaly_alert_task,
)
from backend.core.ws_manager import ConnectionManager # Assuming a WebSocket manager exists

# Configure module logger
logger = logging.getLogger(__name__)

# Initialize the AnomalyDetectionService
anomaly_service = AnomalyDetectionService()

# Initialize WebSocket Connection Manager
manager = ConnectionManager()

# Create router
router = APIRouter(tags=["Anomaly Detection"])


# --- Request/Response Models ---

class AnomalyDetectionRequest(BaseModel):
    entity_id: str
    entity_type: DataEntityType
    strategies: Optional[List[str]] = None
    create_evidence: bool = True

class BulkAnomalyDetectionRequest(BaseModel):
    entities: List[Tuple[str, DataEntityType]]
    strategies: Optional[List[str]] = None
    create_evidence: bool = True

class AnomalyAlertCreate(BaseModel):
    anomaly_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[AnomalySeverity] = None
    tags: Optional[List[str]] = None

class AnomalyAlertUpdate(BaseModel):
    status: AlertStatus
    assigned_to: Optional[str] = None

class AnomalyNotificationRequest(BaseModel):
    notification_channels: List[str]
    custom_message: Optional[str] = None

class DetectionStrategyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    method: DetectionMethod
    entity_types: List[DataEntityType]
    enabled: bool = True
    threshold_config: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class DetectionStrategyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    method: Optional[DetectionMethod] = None
    entity_types: Optional[List[DataEntityType]] = None
    enabled: Optional[bool] = None
    threshold_config: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None

class ContinuousMonitoringRequest(BaseModel):
    scan_interval_minutes: int = 60
    entity_types_to_scan: Optional[List[DataEntityType]] = None
    limit_per_type: int = 100


# --- Anomaly Detection Endpoints ---

@router.post("/anomaly/detect/single", summary="Trigger anomaly detection for a single entity")
async def detect_single_entity(
    request: AnomalyDetectionRequest,
    current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Triggers an asynchronous task to detect anomalies for a single entity.
    """
    try:
        task = detect_single_entity_anomaly_task.apply_async(
            args=[request.entity_id, request.entity_type.value, request.strategies, request.create_evidence]
        )
        return {"message": "Anomaly detection started", "task_id": task.id}
    except Exception as e:
        logger.error(f"Failed to start single entity anomaly detection: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start anomaly detection: {e}"
        )

@router.post("/anomaly/detect/bulk", summary="Trigger anomaly detection for multiple entities")
async def detect_bulk_entities(
    request: BulkAnomalyDetectionRequest,
    current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Triggers an asynchronous task to detect anomalies for multiple entities in bulk.
    """
    try:
        # Convert list of (str, DataEntityType) to (str, str) for Celery task
        entities_for_celery = [(e_id, e_type.value) for e_id, e_type in request.entities]
        task = detect_bulk_anomalies_task.apply_async(
            args=[entities_for_celery, request.strategies, request.create_evidence]
        )
        return {"message": "Bulk anomaly detection started", "task_id": task.id}
    except Exception as e:
        logger.error(f"Failed to start bulk anomaly detection: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start bulk anomaly detection: {e}"
        )

@router.get("/anomaly/results/{result_id}", summary="Get a specific anomaly detection result")
async def get_anomaly_result(
    result_id: str,
    current_user: Any = Depends(get_current_user)
) -> AnomalyDetectionResult:
    """
    Retrieves a specific anomaly detection result by its ID.
    """
    result = await anomaly_service.get_detection_result(result_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly detection result not found")
    return result

@router.get("/anomaly/entities/{entity_type}/{entity_id}/results", summary="Get anomaly detection results for an entity")
async def get_entity_anomaly_results(
    entity_type: DataEntityType,
    entity_id: str,
    limit: int = 100,
    current_user: Any = Depends(get_current_user)
) -> List[AnomalyDetectionResult]:
    """
    Retrieves all anomaly detection results associated with a specific entity.
    """
    results = await anomaly_service.get_entity_results(entity_id, entity_type, limit)
    return results


# --- Alert Management Endpoints ---

@router.post("/anomaly/alerts", summary="Create a new anomaly alert")
async def create_anomaly_alert(
    request: AnomalyAlertCreate,
    current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Creates a new anomaly alert. This can be used to manually create alerts
    or to trigger alerts from external systems.
    """
    try:
        task = generate_anomaly_alert_task.apply_async(
            args=[
                request.anomaly_id,
                request.title,
                request.description,
                request.severity.value if request.severity else None,
                request.tags,
            ]
        )
        return {"message": "Alert creation started", "task_id": task.id}
    except Exception as e:
        logger.error(f"Failed to start alert creation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create alert: {e}"
        )

@router.get("/anomaly/alerts/{alert_id}", summary="Get a specific anomaly alert")
async def get_anomaly_alert(
    alert_id: str,
    current_user: Any = Depends(get_current_user)
) -> AnomalyAlert:
    """
    Retrieves a specific anomaly alert by its ID.
    """
    alert = await anomaly_service.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly alert not found")
    return alert

@router.get("/anomaly/alerts", summary="List anomaly alerts")
async def list_anomaly_alerts(
    status_filter: Optional[AlertStatus] = None,
    severity_filter: Optional[AnomalySeverity] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: Any = Depends(get_current_user)
) -> List[AnomalyAlert]:
    """
    Lists anomaly alerts with optional filtering by status and severity.
    """
    alerts = await anomaly_service.get_alerts(status_filter, severity_filter, limit, offset)
    return alerts

@router.put("/anomaly/alerts/{alert_id}/status", summary="Update the status of an anomaly alert")
async def update_alert_status(
    alert_id: str,
    request: AnomalyAlertUpdate,
    current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Updates the status of an existing anomaly alert.
    """
    try:
        task = update_alert_status_task.apply_async(
            args=[alert_id, request.status.value, request.assigned_to]
        )
        return {"message": "Alert status update started", "task_id": task.id}
    except Exception as e:
        logger.error(f"Failed to start alert status update: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update alert status: {e}"
        )

@router.post("/anomaly/alerts/{alert_id}/notify", summary="Send notifications for an anomaly alert")
async def notify_anomaly_alert(
    alert_id: str,
    request: AnomalyNotificationRequest,
    current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Sends notifications for a specific anomaly alert through configured channels.
    """
    try:
        task = notify_anomaly_alert_task.apply_async(
            args=[alert_id, request.notification_channels, request.custom_message]
        )
        return {"message": "Notification task started", "task_id": task.id}
    except Exception as e:
        logger.error(f"Failed to start notification task: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send notification: {e}"
        )


# --- Detection Strategy Management Endpoints ---

@router.post("/anomaly/strategies", summary="Add a new anomaly detection strategy")
async def add_detection_strategy(
    request: DetectionStrategyCreate,
    current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Adds a new anomaly detection strategy to the system.
    """
    try:
        task = add_detection_strategy_task.apply_async(args=[request.dict()])
        return {"message": "Detection strategy creation started", "task_id": task.id}
    except Exception as e:
        logger.error(f"Failed to start strategy creation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add detection strategy: {e}"
        )

@router.get("/anomaly/strategies", summary="List all anomaly detection strategies")
async def list_detection_strategies(
    current_user: Any = Depends(get_current_user)
) -> List[DetectionStrategy]:
    """
    Lists all configured anomaly detection strategies.
    """
    strategies_dict = await anomaly_service.get_detection_strategies()
    return list(strategies_dict.values())

@router.get("/anomaly/strategies/{strategy_id}", summary="Get a specific anomaly detection strategy")
async def get_detection_strategy(
    strategy_id: str,
    current_user: Any = Depends(get_current_user)
) -> DetectionStrategy:
    """
    Retrieves a specific anomaly detection strategy by its ID.
    """
    strategy = await anomaly_service.get_detection_strategy(strategy_id)
    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection strategy not found")
    return strategy

@router.put("/anomaly/strategies/{strategy_id}", summary="Update an existing anomaly detection strategy")
async def update_detection_strategy(
    strategy_id: str,
    request: DetectionStrategyUpdate,
    current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Updates an existing anomaly detection strategy.
    """
    try:
        task = update_detection_strategy_task.apply_async(args=[strategy_id, request.dict(exclude_unset=True)])
        return {"message": "Detection strategy update started", "task_id": task.id}
    except Exception as e:
        logger.error(f"Failed to start strategy update: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update detection strategy: {e}"
        )

@router.delete("/anomaly/strategies/{strategy_id}", summary="Delete an anomaly detection strategy")
async def delete_detection_strategy(
    strategy_id: str,
    current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Deletes an anomaly detection strategy from the system.
    """
    try:
        task = delete_detection_strategy_task.apply_async(args=[strategy_id])
        return {"message": "Detection strategy deletion started", "task_id": task.id}
    except Exception as e:
        logger.error(f"Failed to start strategy deletion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete detection strategy: {e}"
        )


# --- Real-time Alert Streaming via WebSocket ---

@router.websocket("/anomaly/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time anomaly alert streaming.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Wait for client messages (e.g., filter updates)
            data = await websocket.receive_json()
            
            # Process client message (e.g., update filters)
            filters = data.get("filters", {})
            
            # Send confirmation back to client
            await manager.send_personal_message(
                {"type": "filters_updated", "filters": filters},
                websocket
            )
            
            # Note: Actual alert notifications will be sent by the ConnectionManager
            # when alerts are created or updated in the system
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await manager.disconnect(websocket)


# --- Dashboard Endpoints for Anomaly Statistics ---

@router.get("/anomaly/dashboard/summary", summary="Get overall anomaly summary statistics")
async def get_anomaly_summary(
    days: int = 30,
    current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Provides summary statistics for anomalies detected in the system.
    """
    try:
        # In a real implementation, this would query the database or Redis
        # for actual statistics. For now, we'll return mock data.
        
        return {
            "total_anomalies": 1256,
            "total_alerts": 843,
            "by_severity": {
                "low": 412,
                "medium": 523,
                "high": 245,
                "critical": 76
            },
            "by_type": {
                "statistical_outlier": 345,
                "pattern_match": 567,
                "graph_structure": 234,
                "temporal_pattern": 110
            },
            "by_status": {
                "new": 124,
                "investigating": 56,
                "confirmed": 432,
                "false_positive": 187,
                "resolved": 44
            },
            "time_period_days": days
        }
    except Exception as e:
        logger.error(f"Failed to get anomaly summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get anomaly summary: {e}"
        )

@router.get("/anomaly/dashboard/trends", summary="Get anomaly trends over time")
async def get_anomaly_trends(
    days: int = 30,
    interval: str = "day",  # 'hour', 'day', 'week'
    current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Provides trend data for anomalies over time.
    """
    try:
        # In a real implementation, this would query the database or Redis
        # for actual trend data. For now, we'll return mock data.
        
        # Generate mock time series data
        import random
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if interval == "hour":
            intervals = days * 24
            delta = timedelta(hours=1)
        elif interval == "day":
            intervals = days
            delta = timedelta(days=1)
        elif interval == "week":
            intervals = max(1, days // 7)
            delta = timedelta(weeks=1)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interval: {interval}. Must be 'hour', 'day', or 'week'."
            )
        
        time_points = []
        current_date = start_date
        
        for _ in range(intervals):
            time_points.append(current_date)
            current_date += delta
        
        # Generate mock counts for each anomaly type
        trends = {
            "timestamps": [t.isoformat() for t in time_points],
            "statistical_outlier": [random.randint(0, 20) for _ in range(intervals)],
            "pattern_match": [random.randint(5, 30) for _ in range(intervals)],
            "graph_structure": [random.randint(0, 15) for _ in range(intervals)],
            "temporal_pattern": [random.randint(0, 10) for _ in range(intervals)]
        }
        
        return {
            "trends": trends,
            "time_period_days": days,
            "interval": interval
        }
    except Exception as e:
        logger.error(f"Failed to get anomaly trends: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get anomaly trends: {e}"
        )

@router.get("/anomaly/dashboard/top_anomalies", summary="Get top anomalies by severity")
async def get_top_anomalies(
    limit: int = 10,
    current_user: Any = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Retrieves the top anomalies by severity for dashboard display.
    """
    try:
        # In a real implementation, this would query the database or Redis
        # for actual top anomalies. For now, we'll return mock data.
        
        # Generate mock top anomalies
        import random
        from datetime import datetime, timedelta
        
        anomaly_types = [t.value for t in AnomalyType]
        entity_types = [t.value for t in DataEntityType]
        severities = ["critical", "high"]
        
        top_anomalies = []
        
        for i in range(limit):
            detection_time = datetime.now() - timedelta(minutes=random.randint(0, 60 * 24))
            
            anomaly = {
                "id": f"anom-{uuid.uuid4()}",
                "anomaly_type": random.choice(anomaly_types),
                "severity": random.choice(severities),
                "entity_type": random.choice(entity_types),
                "entity_id": f"0x{random.randint(10**30, 10**40):x}",
                "detection_time": detection_time.isoformat(),
                "score": random.uniform(0.8, 0.99),
                "confidence": random.uniform(0.7, 0.95),
                "has_alert": random.choice([True, False]),
                "has_evidence": random.choice([True, False])
            }
            
            top_anomalies.append(anomaly)
        
        return top_anomalies
    except Exception as e:
        logger.error(f"Failed to get top anomalies: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get top anomalies: {e}"
        )


# --- Background Job Management Endpoints ---

@router.post("/anomaly/jobs/continuous_monitoring", summary="Start continuous anomaly monitoring")
async def start_continuous_monitoring(
    request: ContinuousMonitoringRequest,
    current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Starts a continuous anomaly monitoring job.
    """
    try:
        # Convert entity types to strings for Celery
        entity_types = [et.value for et in request.entity_types] if request.entity_types else None
        
        task = continuous_anomaly_monitoring_task.apply_async(
            args=[request.scan_interval_minutes, entity_types, request.limit_per_type]
        )
        
        return {
            "message": "Continuous monitoring started",
            "task_id": task.id,
            "scan_interval_minutes": request.scan_interval_minutes,
            "entity_types": entity_types,
            "limit_per_type": request.limit_per_type
        }
    except Exception as e:
        logger.error(f"Failed to start continuous monitoring: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start continuous monitoring: {e}"
        )

@router.get("/anomaly/jobs/{task_id}/status", summary="Get status of an anomaly detection task")
async def get_task_status(
    task_id: str,
    current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Retrieves the status of an anomaly detection task.
    """
    try:
        # Get task result from Celery
        from backend.jobs.celery_app import celery_app
        task = celery_app.AsyncResult(task_id)
        
        response = {
            "task_id": task_id,
            "status": task.status,
        }
        
        # Add result if task is successful
        if task.successful():
            response["result"] = task.result
        
        # Add error if task failed
        if task.failed():
            response["error"] = str(task.result)
        
        # Add progress info if available
        if task.info and isinstance(task.info, dict) and "progress" in task.info:
            response["progress"] = task.info["progress"]
            response["step"] = task.info.get("step", "Processing")
        
        return response
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {e}"
        )
