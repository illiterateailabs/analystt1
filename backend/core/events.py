"""
Event system for crew and task progress tracking.

This module provides a comprehensive event system for tracking progress of
crews, agents, and tasks in the Analystt1 platform. It supports multiple
subscribers through a thread-safe event emitter that can broadcast events
to WebSocket connections, Server-Sent Events (SSE), and logging systems.

Example:
    ```python
    # Create an event emitter
    emitter = EventEmitter()
    
    # Subscribe to events
    async def on_task_progress(event):
        print(f"Task {event.task_id} progress: {event.progress}%")
    
    emitter.subscribe(EventType.TASK_PROGRESS, on_task_progress)
    
    # Emit an event
    await emitter.emit(
        EventType.TASK_PROGRESS,
        task_id="123",
        crew_id="456",
        progress=50,
        message="Processing data"
    )
    ```
"""

import asyncio
import enum
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Union

from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)


class EventType(str, enum.Enum):
    """Event types for the event system."""
    
    # Task lifecycle events
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    
    # Agent lifecycle events
    AGENT_STARTED = "agent_started"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    
    # Tool events
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    
    # Crew lifecycle events
    CREW_STARTED = "crew_started"
    CREW_PROGRESS = "crew_progress"
    CREW_COMPLETED = "crew_completed"
    CREW_FAILED = "crew_failed"
    
    # HITL events
    HITL_REVIEW_REQUESTED = "hitl_review_requested"
    HITL_REVIEW_APPROVED = "hitl_review_approved"
    HITL_REVIEW_REJECTED = "hitl_review_rejected"
    
    # System events
    SYSTEM_INFO = "system_info"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_ERROR = "system_error"


@dataclass
class Event:
    """Event data structure with metadata."""
    
    # Event type and identifiers
    type: EventType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Timestamps
    timestamp: float = field(default_factory=time.time)
    timestamp_iso: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Task/Crew/Agent identifiers
    task_id: Optional[str] = None
    crew_id: Optional[str] = None
    agent_id: Optional[str] = None
    tool_id: Optional[str] = None
    
    # Progress information
    progress: Optional[float] = None  # 0-100
    status: Optional[str] = None
    message: Optional[str] = None
    
    # Additional data
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


# Type for event handlers
EventHandler = Callable[[Event], Awaitable[None]]


class EventEmitter:
    """Thread-safe event emitter for broadcasting events to multiple subscribers."""
    
    def __init__(self):
        """Initialize the event emitter."""
        self._handlers: Dict[EventType, List[EventHandler]] = {
            event_type: [] for event_type in EventType
        }
        self._all_handlers: List[EventHandler] = []
        self._lock = asyncio.Lock()
        self._queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the event processing loop."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("Event emitter started")
    
    async def stop(self):
        """Stop the event processing loop."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        logger.info("Event emitter stopped")
    
    async def _process_events(self):
        """Process events from the queue."""
        while self._running:
            try:
                event, handlers = await self._queue.get()
                await self._broadcast_event(event, handlers)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _broadcast_event(self, event: Event, handlers: List[EventHandler]):
        """Broadcast an event to all handlers."""
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    async def emit(self, event_type: EventType, **kwargs) -> Event:
        """
        Emit an event to all subscribers.
        
        Args:
            event_type: Type of the event
            **kwargs: Additional event data
            
        Returns:
            The created event object
        """
        # Create the event
        event = Event(type=event_type, **kwargs)
        
        # Get handlers for this event type and all events
        async with self._lock:
            handlers = self._handlers[event_type].copy() + self._all_handlers.copy()
        
        # Add to queue for processing
        await self._queue.put((event, handlers))
        
        # Log the event
        logger.debug(f"Event emitted: {event.type} - {event.message or ''}")
        
        return event
    
    def subscribe(self, event_type: Optional[EventType], handler: EventHandler) -> Callable[[], None]:
        """
        Subscribe to events of a specific type or all events.
        
        Args:
            event_type: Type of events to subscribe to, or None for all events
            handler: Async function to call when events occur
            
        Returns:
            Unsubscribe function to remove the handler
        """
        async def _add_handler():
            async with self._lock:
                if event_type is None:
                    self._all_handlers.append(handler)
                else:
                    self._handlers[event_type].append(handler)
        
        # Run in the event loop
        asyncio.create_task(_add_handler())
        
        # Return unsubscribe function
        def unsubscribe():
            asyncio.create_task(self._remove_handler(event_type, handler))
        
        return unsubscribe
    
    async def _remove_handler(self, event_type: Optional[EventType], handler: EventHandler):
        """Remove a handler from the subscribers."""
        async with self._lock:
            if event_type is None:
                if handler in self._all_handlers:
                    self._all_handlers.remove(handler)
            else:
                if handler in self._handlers[event_type]:
                    self._handlers[event_type].remove(handler)


# Singleton instance for global use
global_emitter = EventEmitter()


async def emit_event(event_type: EventType, **kwargs) -> Event:
    """
    Emit an event using the global emitter.
    
    This is a convenience function for emitting events without directly
    accessing the global_emitter.
    
    Args:
        event_type: Type of the event
        **kwargs: Additional event data
        
    Returns:
        The created event object
    """
    return await global_emitter.emit(event_type, **kwargs)


def subscribe_to_event(event_type: Optional[EventType], handler: EventHandler) -> Callable[[], None]:
    """
    Subscribe to events using the global emitter.
    
    This is a convenience function for subscribing to events without directly
    accessing the global_emitter.
    
    Args:
        event_type: Type of events to subscribe to, or None for all events
        handler: Async function to call when events occur
        
    Returns:
        Unsubscribe function to remove the handler
    """
    return global_emitter.subscribe(event_type, handler)


async def initialize_events():
    """Initialize the event system."""
    await global_emitter.start()
    logger.info("Event system initialized")


async def shutdown_events():
    """Shutdown the event system."""
    await global_emitter.stop()
    logger.info("Event system shut down")
