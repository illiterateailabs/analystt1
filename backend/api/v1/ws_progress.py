"""
WebSocket API Endpoints for Real-time Progress and Alerts

This module provides FastAPI WebSocket endpoints for real-time communication,
enabling clients to receive:
- General progress updates for all background tasks.
- Task-specific progress updates.
- User-specific progress updates.
- Real-time anomaly alerts.

It integrates with the ConnectionManager to manage WebSocket connections
and broadcast messages, and includes authentication handling.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel

from backend.auth.dependencies import get_current_user
from backend.core.ws_manager import ws_manager # Global instance of ConnectionManager
from backend.jobs.celery_app import celery_app # Import Celery app to monitor tasks

# Configure module logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["WebSockets"])

# --- WebSocket Endpoints ---

@router.websocket("/ws/progress")
async def websocket_general_progress(
    websocket: WebSocket,
    current_user: Any = Depends(get_current_user) # Authenticate WebSocket connection
):
    """
    WebSocket endpoint for general real-time progress updates for all tasks.
    Clients connected to this endpoint will receive all broadcasted messages.
    """
    user_id = current_user.username if current_user else "anonymous"
    await ws_manager.connect(websocket, user_id=user_id)
    try:
        while True:
            # Keep the connection alive. Optionally, receive messages for filtering/control.
            # For general progress, we might not expect client messages often.
            await websocket.receive_text() # Or receive_json() if expecting structured messages
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, user_id=user_id)
        logger.info(f"General progress WebSocket disconnected for {user_id}")
    except Exception as e:
        logger.error(f"Error in general progress WebSocket for {user_id}: {e}", exc_info=True)
        await ws_manager.disconnect(websocket, user_id=user_id)

@router.websocket("/ws/tasks/{task_id}/progress")
async def websocket_task_progress(
    websocket: WebSocket,
    task_id: str,
    current_user: Any = Depends(get_current_user) # Authenticate WebSocket connection
):
    """
    WebSocket endpoint for real-time progress updates for a specific Celery task.
    Clients can subscribe to updates for a given task ID.
    """
    user_id = current_user.username if current_user else "anonymous"
    await ws_manager.connect(websocket, user_id=user_id)
    await ws_manager.subscribe_to_task_progress(websocket, task_id)
    try:
        while True:
            # Keep the connection alive. Clients might send messages to unsubscribe or
            # request specific updates, but for now, just keep alive.
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.unsubscribe_from_task_progress(websocket, task_id)
        ws_manager.disconnect(websocket, user_id=user_id)
        logger.info(f"Task progress WebSocket disconnected for task {task_id} by {user_id}")
    except Exception as e:
        logger.error(f"Error in task progress WebSocket for task {task_id} by {user_id}: {e}", exc_info=True)
        await ws_manager.unsubscribe_from_task_progress(websocket, task_id)
        await ws_manager.disconnect(websocket, user_id=user_id)

@router.websocket("/ws/users/{user_id}/progress")
async def websocket_user_progress(
    websocket: WebSocket,
    user_id: str,
    current_user: Any = Depends(get_current_user) # Authenticate WebSocket connection
):
    """
    WebSocket endpoint for real-time progress updates specific to a user.
    Clients can subscribe to updates relevant to their user ID.
    """
    # Ensure the authenticated user matches the requested user_id, or is an admin
    if current_user and current_user.username != user_id and not current_user.is_admin:
        logger.warning(f"Unauthorized attempt to access user {user_id}'s WebSocket by {current_user.username}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await ws_manager.connect(websocket, user_id=user_id) # Connects to the user's room
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, user_id=user_id)
        logger.info(f"User-specific progress WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"Error in user-specific progress WebSocket for user {user_id}: {e}", exc_info=True)
        await ws_manager.disconnect(websocket, user_id=user_id)

@router.websocket("/ws/alerts")
async def websocket_alerts(
    websocket: WebSocket,
    current_user: Any = Depends(get_current_user) # Authenticate WebSocket connection
):
    """
    WebSocket endpoint for real-time anomaly alert streaming.
    Clients connected to this endpoint will receive all anomaly alerts.
    """
    user_id = current_user.username if current_user else "anonymous"
    await ws_manager.connect(websocket, user_id=user_id)
    try:
        while True:
            # Clients might send messages to filter alerts, etc.
            # For now, just keep the connection alive.
            data = await websocket.receive_json()
            logger.debug(f"Received message from alert WebSocket ({user_id}): {data}")
            # Example: if client sends filter updates, process them here
            # await ws_manager.send_personal_message({"status": "filters_received"}, websocket)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, user_id=user_id)
        logger.info(f"Alerts WebSocket disconnected for {user_id}")
    except Exception as e:
        logger.error(f"Error in alerts WebSocket for {user_id}: {e}", exc_info=True)
        await ws_manager.disconnect(websocket, user_id=user_id)

# --- Startup/Shutdown Events for Celery Monitor ---

@router.on_event("startup")
async def startup_websocket_manager_celery_monitor():
    """
    FastAPI startup event to launch the Celery task monitor in the background.
    """
    # Ensure celery_app is imported and available
    ws_manager.start_celery_monitor(celery_app)
    logger.info("WebSocket manager's Celery monitor started.")

@router.on_event("shutdown")
async def shutdown_websocket_manager_celery_monitor():
    """
    FastAPI shutdown event to stop the Celery task monitor gracefully.
    """
    ws_manager.stop_celery_monitor()
    logger.info("WebSocket manager's Celery monitor stopped.")
