"""
WebSocket Connection Manager

This module provides a comprehensive WebSocket connection manager for handling
real-time communication, particularly for sending progress updates for Celery tasks
and broadcasting messages to connected clients.

Features:
- Manages client connections (connect/disconnect).
- Supports room-based subscriptions (e.g., per user, per task ID).
- Broadcasts messages to all or specific subscribed clients.
- Integrates with Celery to push real-time task progress updates.
- Includes basic error handling for WebSocket disconnections.
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Set, Optional

from fastapi import WebSocket, WebSocketDisconnect

# Configure module logger
logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections and handles real-time message broadcasting.
    """
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        # Rooms for targeted broadcasting, e.g., {user_id: {websocket1, websocket2}}
        self.rooms: defaultdict[str, Set[WebSocket]] = defaultdict(set)
        self._celery_monitor_task: Optional[asyncio.Task] = None
        logger.info("WebSocket ConnectionManager initialized.")

    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None):
        """
        Establishes a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection to establish
            user_id: Optional user ID for user-specific rooms
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        if user_id:
            self.rooms[user_id].add(websocket)
            logger.info(f"WebSocket connected for user {user_id}. Total connections: {len(self.active_connections)}")
        else:
            logger.info(f"WebSocket connected (anonymous). Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, user_id: Optional[str] = None):
        """
        Closes a WebSocket connection.
        
        Args:
            websocket: The WebSocket connection to close
            user_id: Optional user ID for cleaning up user-specific rooms
        """
        self.active_connections.discard(websocket)
        if user_id and user_id in self.rooms:
            self.rooms[user_id].discard(websocket)
            if not self.rooms[user_id]:
                del self.rooms[user_id] # Clean up empty rooms
            logger.info(f"WebSocket disconnected for user {user_id}. Remaining connections: {len(self.active_connections)}")
        else:
            logger.info(f"WebSocket disconnected (anonymous). Remaining connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """
        Sends a message to a specific WebSocket connection.
        
        Args:
            message: The message to send
            websocket: The WebSocket connection to send to
        """
        try:
            await websocket.send_json(message)
        except WebSocketDisconnect:
            self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error sending personal message to WebSocket: {e}", exc_info=True)

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcasts a message to all active WebSocket connections.
        
        Args:
            message: The message to broadcast
        """
        disconnected_websockets = []
        for connection in list(self.active_connections): # Iterate over a copy
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected_websockets.append(connection)
            except Exception as e:
                logger.error(f"Error broadcasting message to WebSocket {connection}: {e}", exc_info=True)
                disconnected_websockets.append(connection)
        
        for ws in disconnected_websockets:
            self.disconnect(ws)
        logger.debug(f"Broadcasted message to {len(self.active_connections)} connections.")

    async def send_to_room(self, room_id: str, message: Dict[str, Any]):
        """
        Sends a message to all clients subscribed to a specific room.
        
        Args:
            room_id: ID of the room to send to
            message: The message to send
        """
        if room_id not in self.rooms:
            logger.warning(f"Room '{room_id}' has no active subscribers.")
            return

        disconnected_websockets = []
        for connection in list(self.rooms[room_id]): # Iterate over a copy
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected_websockets.append(connection)
            except Exception as e:
                logger.error(f"Error sending message to room '{room_id}' for WebSocket {connection}: {e}", exc_info=True)
                disconnected_websockets.append(connection)
        
        for ws in disconnected_websockets:
            self.rooms[room_id].discard(ws)
            self.active_connections.discard(ws) # Also remove from overall active connections
        
        if not self.rooms[room_id]:
            del self.rooms[room_id]
        logger.debug(f"Sent message to room '{room_id}'. Remaining subscribers: {len(self.rooms.get(room_id, []))}")

    def start_celery_monitor(self, celery_app_instance: Any, check_interval_seconds: int = 1):
        """
        Starts a background task to monitor Celery task progress and send updates.
        
        Args:
            celery_app_instance: Celery app instance to monitor
            check_interval_seconds: How often to check for updates (in seconds)
        """
        if self._celery_monitor_task and not self._celery_monitor_task.done():
            logger.warning("Celery monitor task is already running.")
            return

        self._celery_monitor_task = asyncio.create_task(
            self._monitor_celery_tasks(celery_app_instance, check_interval_seconds)
        )
        logger.info("Celery task monitor background task started.")

    def stop_celery_monitor(self):
        """
        Stops the background Celery task monitor.
        """
        if self._celery_monitor_task:
            self._celery_monitor_task.cancel()
            self._celery_monitor_task = None
            logger.info("Celery task monitor background task stopped.")

    async def _monitor_celery_tasks(self, celery_app_instance: Any, check_interval_seconds: int):
        """
        Internal method to poll Celery for task status and send updates.
        
        Args:
            celery_app_instance: Celery app instance to monitor
            check_interval_seconds: How often to check for updates (in seconds)
        """
        logger.info(f"Monitoring Celery tasks with interval: {check_interval_seconds}s")
        # Keep track of tasks we've already sent final status for
        completed_tasks: Set[str] = set()

        while True:
            try:
                # Get all active tasks (this might be resource intensive for very large systems)
                # For a more scalable solution, consider Celery events or a dedicated result backend listener
                i = celery_app_instance.control.inspect()
                active_tasks = i.active()
                reserved_tasks = i.reserved()
                scheduled_tasks = i.scheduled()

                all_task_ids = set()
                if active_tasks:
                    for worker, tasks in active_tasks.items():
                        for task in tasks:
                            all_task_ids.add(task['id'])
                if reserved_tasks:
                    for worker, tasks in reserved_tasks.items():
                        for task in tasks:
                            all_task_ids.add(task['id'])
                if scheduled_tasks:
                    for worker, tasks in scheduled_tasks.items():
                        for task in tasks:
                            all_task_ids.add(task['id'])

                for task_id in all_task_ids:
                    if task_id in completed_tasks:
                        continue # Already processed final status

                    task_result = celery_app_instance.AsyncResult(task_id)
                    
                    status_message = {
                        "type": "celery_task_update",
                        "task_id": task_id,
                        "status": task_result.state,
                        "info": task_result.info,
                        "timestamp": datetime.now().isoformat()
                    }

                    # Send update to a room specific to this task, and potentially a user's room
                    await self.send_to_room(f"task_{task_id}", status_message)
                    
                    # If task is done (SUCCESS, FAILURE, REVOKED, etc.), mark as completed
                    if task_result.ready():
                        completed_tasks.add(task_id)
                        logger.debug(f"Task {task_id} completed with status {task_result.state}")

            except asyncio.CancelledError:
                logger.info("Celery monitor task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in Celery monitor loop: {e}", exc_info=True)
            
            await asyncio.sleep(check_interval_seconds)

    async def subscribe_to_task_progress(self, websocket: WebSocket, task_id: str):
        """
        Allows a WebSocket client to subscribe to progress updates for a specific task.
        
        Args:
            websocket: The WebSocket connection
            task_id: ID of the task to subscribe to
        """
        room_id = f"task_{task_id}"
        self.rooms[room_id].add(websocket)
        logger.info(f"WebSocket {websocket.client} subscribed to task {task_id} progress.")
        # Optionally send current status immediately
        # task_result = celery_app.AsyncResult(task_id)
        # await self.send_personal_message({
        #     "type": "celery_task_update",
        #     "task_id": task_id,
        #     "status": task_result.state,
        #     "info": task_result.info,
        #     "timestamp": datetime.now().isoformat()
        # }, websocket)

    async def unsubscribe_from_task_progress(self, websocket: WebSocket, task_id: str):
        """
        Removes a WebSocket client's subscription from a task's progress updates.
        
        Args:
            websocket: The WebSocket connection
            task_id: ID of the task to unsubscribe from
        """
        room_id = f"task_{task_id}"
        if room_id in self.rooms:
            self.rooms[room_id].discard(websocket)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
            logger.info(f"WebSocket {websocket.client} unsubscribed from task {task_id} progress.")

    async def send_task_progress_update(self, task_id: str, status: str, progress: int, info: Dict[str, Any] = None):
        """
        Sends a task progress update to all subscribed clients.
        
        Args:
            task_id: ID of the task
            status: Current status of the task
            progress: Progress percentage (0-100)
            info: Additional information about the task
        """
        room_id = f"task_{task_id}"
        message = {
            "type": "task_progress",
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "info": info or {},
            "timestamp": datetime.now().isoformat()
        }
        await self.send_to_room(room_id, message)
        logger.debug(f"Sent progress update for task {task_id}: {status} ({progress}%)")

    async def send_alert(self, alert_type: str, message: str, severity: str = "info", data: Dict[str, Any] = None):
        """
        Broadcasts an alert message to all connected clients.
        
        Args:
            alert_type: Type of alert (e.g., "anomaly", "system", "error")
            message: Alert message text
            severity: Alert severity ("info", "warning", "error", "critical")
            data: Additional data related to the alert
        """
        alert_message = {
            "type": "alert",
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(alert_message)
        logger.info(f"Broadcast {severity} alert: {message}")


# Global instance of the ConnectionManager
# This should be initialized once at application startup
ws_manager = ConnectionManager()

# Example of how to start the monitor (e.g., in FastAPI startup event)
# from backend.jobs.celery_app import celery_app # Import your celery app instance
# @app.on_event("startup")
# async def startup_websocket_manager():
#     ws_manager.start_celery_monitor(celery_app)

# @app.on_event("shutdown")
# async def shutdown_websocket_manager():
#     ws_manager.stop_celery_monitor()
