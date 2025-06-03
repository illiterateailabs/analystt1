"""
WebSocket endpoint for real-time task progress updates.

This module provides a WebSocket endpoint for clients to receive real-time
updates about task progress, agent activities, and other events related to
a specific analysis task.

The endpoint authenticates users via JWT tokens and subscribes to events
from the global event emitter, filtering them to only send events relevant
to the requested task.

Example:
    ```javascript
    // Client-side JavaScript
    const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/tasks/${taskId}?token=${jwt}`);
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(`Task ${data.task_id} update: ${data.message}`);
    };
    ```
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set, Any, Callable

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query, status
from fastapi.security import APIKeyHeader
from jose import JWTError, jwt
from starlette.websockets import WebSocketState

from backend.auth.dependencies import get_current_user, oauth2_scheme
from backend.auth.jwt_handler import decode_token, verify_token
from backend.config import settings
from backend.core.events import EventType, Event, global_emitter, subscribe_to_event
from backend.core.logging import get_logger

# Configure logging
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/ws", tags=["websockets"])

# Store active connections
active_connections: Dict[str, Set[WebSocket]] = {}

# Heartbeat interval in seconds
HEARTBEAT_INTERVAL = 30


async def authenticate_websocket(websocket: WebSocket) -> Optional[Dict[str, Any]]:
    """
    Authenticate a WebSocket connection using JWT.
    
    Args:
        websocket: The WebSocket connection
        
    Returns:
        The decoded token payload if authentication is successful, None otherwise
    """
    # Get token from query params or headers
    token = websocket.query_params.get("token")
    
    if not token:
        # Try to get from headers
        auth_header = websocket.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
    
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing authentication token")
        return None
    
    try:
        # Verify and decode the token
        payload = decode_token(token)
        if not payload:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
            return None
        
        return payload
    except JWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
        return None


async def send_heartbeat(websocket: WebSocket):
    """
    Send periodic heartbeats to keep the connection alive.
    
    Args:
        websocket: The WebSocket connection
    """
    while True:
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({"type": "heartbeat", "timestamp": asyncio.get_event_loop().time()}))
            else:
                break
            await asyncio.sleep(HEARTBEAT_INTERVAL)
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
            break


async def handle_event(event: Event, websocket: WebSocket, task_id: str):
    """
    Handle an event and send it to the WebSocket if relevant.
    
    Args:
        event: The event to handle
        websocket: The WebSocket connection
        task_id: The task ID to filter events for
    """
    # Only send events for the specific task
    if event.task_id != task_id:
        return
    
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(event.to_json())
    except Exception as e:
        logger.error(f"Error sending event to WebSocket: {e}")


@router.websocket("/tasks/{task_id}")
async def websocket_task_progress(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time task progress updates.
    
    Args:
        websocket: The WebSocket connection
        task_id: The ID of the task to subscribe to
    """
    # Accept the connection
    await websocket.accept()
    
    # Authenticate the user
    payload = await authenticate_websocket(websocket)
    if not payload:
        return  # Connection was closed by authenticate_websocket
    
    # Store the connection
    if task_id not in active_connections:
        active_connections[task_id] = set()
    active_connections[task_id].add(websocket)
    
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
    
    try:
        # Subscribe to events for this task
        async def on_event(event: Event):
            await handle_event(event, websocket, task_id)
        
        # Subscribe to all event types
        unsubscribe = subscribe_to_event(None, on_event)
        
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "task_id": task_id,
            "message": "Connected to task progress feed",
            "timestamp": asyncio.get_event_loop().time()
        }))
        
        # Wait for messages from the client (e.g., pong responses)
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle pong responses or client commands
            if message.get("type") == "pong":
                logger.debug(f"Received pong from client for task {task_id}")
            elif message.get("type") == "subscribe":
                # Client can subscribe to specific event types
                logger.debug(f"Client subscribed to events for task {task_id}")
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
    finally:
        # Clean up
        if task_id in active_connections and websocket in active_connections[task_id]:
            active_connections[task_id].remove(websocket)
            if not active_connections[task_id]:
                del active_connections[task_id]
        
        # Cancel heartbeat task
        heartbeat_task.cancel()
        
        # Unsubscribe from events
        if 'unsubscribe' in locals():
            unsubscribe()


@router.websocket("/tasks")
async def websocket_all_tasks(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates for all tasks.
    
    This endpoint is intended for admin dashboards that need to monitor
    all tasks in the system.
    
    Args:
        websocket: The WebSocket connection
    """
    # Accept the connection
    await websocket.accept()
    
    # Authenticate the user
    payload = await authenticate_websocket(websocket)
    if not payload:
        return  # Connection was closed by authenticate_websocket
    
    # Check if user has admin role
    if "role" not in payload or payload["role"] != "admin":
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Insufficient permissions")
        return
    
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
    
    try:
        # Subscribe to all events
        async def on_all_events(event: Event):
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(event.to_json())
            except Exception as e:
                logger.error(f"Error sending event to admin WebSocket: {e}")
        
        # Subscribe to all event types
        unsubscribe = subscribe_to_event(None, on_all_events)
        
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Connected to all tasks progress feed",
            "timestamp": asyncio.get_event_loop().time()
        }))
        
        # Wait for messages from the client
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        logger.info("Admin WebSocket disconnected")
    except Exception as e:
        logger.error(f"Admin WebSocket error: {e}")
    finally:
        # Cancel heartbeat task
        heartbeat_task.cancel()
        
        # Unsubscribe from events
        if 'unsubscribe' in locals():
            unsubscribe()


# Utility functions for broadcasting to all clients for a task

async def broadcast_to_task(task_id: str, message: Dict[str, Any]):
    """
    Broadcast a message to all clients subscribed to a task.
    
    Args:
        task_id: The task ID
        message: The message to broadcast
    """
    if task_id not in active_connections:
        return
    
    disconnected = set()
    message_json = json.dumps(message)
    
    for websocket in active_connections[task_id]:
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(message_json)
        except Exception:
            disconnected.add(websocket)
    
    # Clean up disconnected clients
    for websocket in disconnected:
        active_connections[task_id].remove(websocket)
    
    if not active_connections[task_id]:
        del active_connections[task_id]


# Initialize the WebSocket routes
def init_websocket_routes(app):
    """
    Initialize WebSocket routes in the FastAPI application.
    
    Args:
        app: The FastAPI application
    """
    app.include_router(router)
    logger.info("WebSocket routes initialized")
