"""
Event system for the application.

This module provides a comprehensive typed event system with:
- Typed event classes (GraphAddEvent, CacheAddEvent, LLMUsageEvent, etc.)
- @subscribe decorator for registering event handlers
- EventBus for managing subscriptions and publishing
- Async event handling support
- Event persistence and replay capabilities
- Integration with logging and metrics
- Event filtering and routing
- Support for event aggregation and batching
"""

import asyncio
import functools
import inspect
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, ClassVar, Dict, Generic, List, Optional, Set, 
    Type, TypeVar, Union, cast, get_type_hints
)

import redis
from pydantic import BaseModel, Field

# Configure module logger
logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar("T")
EventType = TypeVar("EventType", bound="Event")
HandlerType = Callable[[EventType], None]
AsyncHandlerType = Callable[[EventType], asyncio.coroutine]

# Global event bus instance
_event_bus: Optional["EventBus"] = None


class EventPriority(Enum):
    """Priority levels for event handling."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventCategory(str, Enum):
    """Categories of events for filtering and routing."""
    SYSTEM = "system"
    DATA = "data"
    USER = "user"
    SECURITY = "security"
    ANALYTICS = "analytics"
    INTEGRATION = "integration"
    ERROR = "error"


@dataclass
class Event:
    """
    Base class for all events in the system.
    
    All event types should inherit from this class and add their specific fields.
    """
    # Common fields for all events
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    category: EventCategory = EventCategory.SYSTEM
    priority: EventPriority = EventPriority.NORMAL
    
    # Class variable to store event type name
    event_type: ClassVar[str] = "event"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = asdict(self)
        data["event_type"] = self.event_type
        data["timestamp"] = self.timestamp.isoformat()
        data["category"] = self.category.value
        data["priority"] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        # Convert string timestamp back to datetime
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        # Convert string category and priority back to enum
        if "category" in data and isinstance(data["category"], str):
            data["category"] = EventCategory(data["category"])
        
        if "priority" in data and isinstance(data["priority"], (int, str)):
            if isinstance(data["priority"], str):
                data["priority"] = int(data["priority"])
            data["priority"] = EventPriority(data["priority"])
        
        # Remove event_type from data as it's a class variable
        if "event_type" in data:
            del data["event_type"]
        
        return cls(**data)


# System Events
@dataclass
class SystemStartEvent(Event):
    """Event fired when the system starts."""
    event_type: ClassVar[str] = "system.start"
    version: str = "1.8.0-beta"
    environment: str = "development"


@dataclass
class SystemShutdownEvent(Event):
    """Event fired when the system shuts down."""
    event_type: ClassVar[str] = "system.shutdown"
    reason: Optional[str] = None


@dataclass
class ConfigChangeEvent(Event):
    """Event fired when configuration changes."""
    event_type: ClassVar[str] = "system.config_change"
    config_key: str = ""
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None


# Data Events
@dataclass
class GraphAddEvent(Event):
    """Event fired when data is added to the graph database."""
    event_type: ClassVar[str] = "data.graph_add"
    category: EventCategory = EventCategory.DATA
    node_count: int = 0
    relationship_count: int = 0
    node_types: List[str] = field(default_factory=list)
    relationship_types: List[str] = field(default_factory=list)
    query_time_ms: float = 0.0
    source: str = ""
    chain: Optional[str] = None


@dataclass
class CacheAddEvent(Event):
    """Event fired when data is added to the cache."""
    event_type: ClassVar[str] = "data.cache_add"
    category: EventCategory = EventCategory.DATA
    key: str = ""
    ttl_seconds: int = 0
    size_bytes: int = 0
    cache_type: str = "default"  # default, vector, etc.


@dataclass
class CacheHitEvent(Event):
    """Event fired when there's a cache hit."""
    event_type: ClassVar[str] = "data.cache_hit"
    category: EventCategory = EventCategory.DATA
    key: str = ""
    cache_type: str = "default"
    age_seconds: float = 0.0


@dataclass
class CacheMissEvent(Event):
    """Event fired when there's a cache miss."""
    event_type: ClassVar[str] = "data.cache_miss"
    category: EventCategory = EventCategory.DATA
    key: str = ""
    cache_type: str = "default"


# Integration Events
@dataclass
class ApiRequestEvent(Event):
    """Event fired when an API request is made."""
    event_type: ClassVar[str] = "integration.api_request"
    category: EventCategory = EventCategory.INTEGRATION
    provider_id: str = ""
    endpoint: str = ""
    method: str = "GET"
    status_code: Optional[int] = None
    duration_ms: float = 0.0
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    cache_hit: bool = False
    retries: int = 0
    error: Optional[str] = None


@dataclass
class LLMUsageEvent(Event):
    """Event fired when an LLM is used."""
    event_type: ClassVar[str] = "integration.llm_usage"
    category: EventCategory = EventCategory.INTEGRATION
    model: str = ""
    operation: str = ""  # completion, embedding, etc.
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    prompt_template: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class DatabaseOperationEvent(Event):
    """Event fired when a database operation is performed."""
    event_type: ClassVar[str] = "integration.db_operation"
    category: EventCategory = EventCategory.INTEGRATION
    database: str = ""  # postgres, neo4j, redis, etc.
    operation: str = ""  # query, insert, update, delete, etc.
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    rows_affected: Optional[int] = None


# Analytics Events
@dataclass
class AgentExecutionEvent(Event):
    """Event fired when an agent executes a task."""
    event_type: ClassVar[str] = "analytics.agent_execution"
    category: EventCategory = EventCategory.ANALYTICS
    agent_type: str = ""
    task: str = ""
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    memory_usage_bytes: Optional[int] = None
    tokens_used: Optional[int] = None


@dataclass
class FraudDetectionEvent(Event):
    """Event fired when fraud is detected."""
    event_type: ClassVar[str] = "analytics.fraud_detection"
    category: EventCategory = EventCategory.ANALYTICS
    detection_type: str = ""
    chain: str = ""
    severity: str = "medium"
    confidence: float = 0.0
    entity_ids: List[str] = field(default_factory=list)
    detection_method: str = ""
    evidence: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisTaskEvent(Event):
    """Event fired when an analysis task is performed."""
    event_type: ClassVar[str] = "analytics.analysis_task"
    category: EventCategory = EventCategory.ANALYTICS
    analysis_type: str = ""
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    result_summary: Optional[str] = None


# User Events
@dataclass
class UserAuthEvent(Event):
    """Event fired for user authentication events."""
    event_type: ClassVar[str] = "user.auth"
    category: EventCategory = EventCategory.SECURITY
    user_id: str = ""
    action: str = ""  # login, logout, register, etc.
    success: bool = True
    error: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class UserActionEvent(Event):
    """Event fired when a user performs an action."""
    event_type: ClassVar[str] = "user.action"
    category: EventCategory = EventCategory.USER
    user_id: str = ""
    action: str = ""
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Error Events
@dataclass
class ErrorEvent(Event):
    """Event fired when an error occurs."""
    event_type: ClassVar[str] = "error"
    category: EventCategory = EventCategory.ERROR
    priority: EventPriority = EventPriority.HIGH
    error_type: str = ""
    message: str = ""
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class EventFilter:
    """
    Filter for events based on various criteria.
    
    This class is used to filter events based on type, category, priority,
    timestamp range, and custom predicates.
    """
    
    def __init__(
        self,
        event_types: Optional[List[str]] = None,
        categories: Optional[List[EventCategory]] = None,
        min_priority: Optional[EventPriority] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        custom_filter: Optional[Callable[[Event], bool]] = None,
    ):
        """
        Initialize an event filter.
        
        Args:
            event_types: List of event types to include
            categories: List of event categories to include
            min_priority: Minimum priority level to include
            start_time: Include events after this time
            end_time: Include events before this time
            custom_filter: Custom predicate function for filtering
        """
        self.event_types = event_types
        self.categories = categories
        self.min_priority = min_priority
        self.start_time = start_time
        self.end_time = end_time
        self.custom_filter = custom_filter
    
    def matches(self, event: Event) -> bool:
        """
        Check if an event matches this filter.
        
        Args:
            event: The event to check
            
        Returns:
            True if the event matches, False otherwise
        """
        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        # Check category
        if self.categories and event.category not in self.categories:
            return False
        
        # Check priority
        if self.min_priority and event.priority.value < self.min_priority.value:
            return False
        
        # Check timestamp range
        if self.start_time and event.timestamp < self.start_time:
            return False
        
        if self.end_time and event.timestamp > self.end_time:
            return False
        
        # Apply custom filter if provided
        if self.custom_filter and not self.custom_filter(event):
            return False
        
        return True


class EventHandler:
    """
    Wrapper for event handler functions with metadata.
    
    This class wraps an event handler function with additional metadata
    such as the event type it handles, its priority, and whether it's async.
    """
    
    def __init__(
        self,
        handler: Union[HandlerType, AsyncHandlerType],
        event_type: str,
        priority: EventPriority = EventPriority.NORMAL,
        is_async: bool = False,
        filter_: Optional[EventFilter] = None,
    ):
        """
        Initialize an event handler.
        
        Args:
            handler: The handler function
            event_type: The event type this handler is for
            priority: The priority of this handler
            is_async: Whether the handler is async
            filter_: Additional filter for events
        """
        self.handler = handler
        self.event_type = event_type
        self.priority = priority
        self.is_async = is_async
        self.filter = filter_
    
    def matches(self, event: Event) -> bool:
        """
        Check if this handler should process the given event.
        
        Args:
            event: The event to check
            
        Returns:
            True if this handler should process the event, False otherwise
        """
        # Check event type
        if self.event_type != event.event_type:
            return False
        
        # Apply additional filter if provided
        if self.filter and not self.filter.matches(event):
            return False
        
        return True
    
    async def call_async(self, event: Event) -> None:
        """
        Call this handler asynchronously.
        
        Args:
            event: The event to handle
        """
        if self.is_async:
            # Handler is already async
            await self.handler(event)
        else:
            # Run sync handler in a thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.handler, event)
    
    def call(self, event: Event) -> None:
        """
        Call this handler synchronously.
        
        Args:
            event: The event to handle
        """
        if self.is_async:
            # Create a new event loop for the async handler
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.handler(event))
            finally:
                loop.close()
        else:
            # Call sync handler directly
            self.handler(event)


class EventBatch:
    """
    Batch of events for efficient processing.
    
    This class collects events of the same type for batch processing.
    """
    
    def __init__(self, event_type: str, max_size: int = 100, max_age_seconds: float = 5.0):
        """
        Initialize an event batch.
        
        Args:
            event_type: The type of events in this batch
            max_size: Maximum number of events in the batch
            max_age_seconds: Maximum age of the batch before it's processed
        """
        self.event_type = event_type
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.events: List[Event] = []
        self.created_at = time.time()
    
    def add(self, event: Event) -> bool:
        """
        Add an event to the batch.
        
        Args:
            event: The event to add
            
        Returns:
            True if the batch is full after adding, False otherwise
        """
        if event.event_type != self.event_type:
            raise ValueError(f"Event type mismatch: {event.event_type} != {self.event_type}")
        
        self.events.append(event)
        return len(self.events) >= self.max_size
    
    def is_ready(self) -> bool:
        """
        Check if the batch is ready for processing.
        
        Returns:
            True if the batch is ready, False otherwise
        """
        return (
            len(self.events) >= self.max_size or
            time.time() - self.created_at >= self.max_age_seconds
        )
    
    def clear(self) -> None:
        """Clear the batch."""
        self.events.clear()
        self.created_at = time.time()


class EventPersistence(ABC):
    """
    Abstract base class for event persistence.
    
    Implementations of this class handle persisting events to storage
    and retrieving them for replay.
    """
    
    @abstractmethod
    def store_event(self, event: Event) -> None:
        """
        Store an event.
        
        Args:
            event: The event to store
        """
        pass
    
    @abstractmethod
    def get_events(
        self,
        filter_: Optional[EventFilter] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Event]:
        """
        Get events from storage.
        
        Args:
            filter_: Filter for events
            limit: Maximum number of events to return
            offset: Offset for pagination
            
        Returns:
            List of events
        """
        pass
    
    @abstractmethod
    def get_event_count(self, filter_: Optional[EventFilter] = None) -> int:
        """
        Get the count of events matching a filter.
        
        Args:
            filter_: Filter for events
            
        Returns:
            Count of matching events
        """
        pass
    
    @abstractmethod
    def clear_events(self, filter_: Optional[EventFilter] = None) -> int:
        """
        Clear events from storage.
        
        Args:
            filter_: Filter for events to clear
            
        Returns:
            Number of events cleared
        """
        pass


class RedisEventPersistence(EventPersistence):
    """
    Redis implementation of event persistence.
    
    This class stores events in Redis for persistence and replay.
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "events",
        max_events: int = 10000,
    ):
        """
        Initialize Redis event persistence.
        
        Args:
            redis_client: Redis client
            key_prefix: Prefix for Redis keys
            max_events: Maximum number of events to store
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.max_events = max_events
        self.event_list_key = f"{key_prefix}:list"
        self.event_types_key = f"{key_prefix}:types"
    
    def store_event(self, event: Event) -> None:
        """
        Store an event in Redis.
        
        Args:
            event: The event to store
        """
        try:
            # Convert event to JSON
            event_json = json.dumps(event.to_dict())
            
            # Store the event with its ID as key
            event_key = f"{self.key_prefix}:{event.id}"
            self.redis.set(event_key, event_json)
            
            # Add to the event list (sorted by timestamp)
            score = event.timestamp.timestamp()
            self.redis.zadd(self.event_list_key, {event.id: score})
            
            # Add to the event type set
            self.redis.sadd(self.event_types_key, event.event_type)
            
            # Add to the event type list (for faster filtering)
            type_list_key = f"{self.key_prefix}:type:{event.event_type}"
            self.redis.zadd(type_list_key, {event.id: score})
            
            # Trim the event list if it's too large
            if self.redis.zcard(self.event_list_key) > self.max_events:
                # Get the oldest events to remove
                to_remove = self.redis.zrange(
                    self.event_list_key,
                    0,
                    self.redis.zcard(self.event_list_key) - self.max_events - 1,
                )
                
                # Remove from the main list
                if to_remove:
                    self.redis.zrem(self.event_list_key, *to_remove)
                    
                    # Remove the event data
                    for event_id in to_remove:
                        event_key = f"{self.key_prefix}:{event_id.decode()}"
                        # Get the event to find its type
                        event_data = self.redis.get(event_key)
                        if event_data:
                            try:
                                event_dict = json.loads(event_data)
                                event_type = event_dict.get("event_type")
                                if event_type:
                                    # Remove from the type list
                                    type_list_key = f"{self.key_prefix}:type:{event_type}"
                                    self.redis.zrem(type_list_key, event_id)
                            except json.JSONDecodeError:
                                pass
                        
                        # Remove the event data
                        self.redis.delete(event_key)
        
        except Exception as e:
            logger.error(f"Error storing event in Redis: {e}")
    
    def get_events(
        self,
        filter_: Optional[EventFilter] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Event]:
        """
        Get events from Redis.
        
        Args:
            filter_: Filter for events
            limit: Maximum number of events to return
            offset: Offset for pagination
            
        Returns:
            List of events
        """
        try:
            events = []
            
            # Determine which key to use based on filter
            if filter_ and filter_.event_types and len(filter_.event_types) == 1:
                # If filtering by a single event type, use the type-specific list
                event_type = filter_.event_types[0]
                key = f"{self.key_prefix}:type:{event_type}"
            else:
                # Otherwise, use the main event list
                key = self.event_list_key
            
            # Get event IDs from the sorted set
            if filter_ and filter_.start_time and filter_.end_time:
                # Filter by time range
                min_score = filter_.start_time.timestamp()
                max_score = filter_.end_time.timestamp()
                event_ids = self.redis.zrangebyscore(
                    key,
                    min_score,
                    max_score,
                    start=offset,
                    num=limit,
                )
            else:
                # Get all events with pagination
                event_ids = self.redis.zrange(
                    key,
                    offset,
                    offset + (limit - 1 if limit else -1),
                )
            
            # Get event data for each ID
            for event_id in event_ids:
                event_key = f"{self.key_prefix}:{event_id.decode()}"
                event_data = self.redis.get(event_key)
                
                if event_data:
                    try:
                        event_dict = json.loads(event_data)
                        event_type = event_dict.get("event_type")
                        
                        # Create the appropriate event object based on type
                        event_class = _get_event_class_by_type(event_type)
                        if event_class:
                            event = event_class.from_dict(event_dict)
                            
                            # Apply additional filtering
                            if filter_ and not filter_.matches(event):
                                continue
                            
                            events.append(event)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in event data: {event_data}")
                    except Exception as e:
                        logger.warning(f"Error creating event object: {e}")
            
            return events
        
        except Exception as e:
            logger.error(f"Error getting events from Redis: {e}")
            return []
    
    def get_event_count(self, filter_: Optional[EventFilter] = None) -> int:
        """
        Get the count of events matching a filter.
        
        Args:
            filter_: Filter for events
            
        Returns:
            Count of matching events
        """
        try:
            # If no filter, return the total count
            if not filter_:
                return self.redis.zcard(self.event_list_key)
            
            # If filtering by event type only, use the type-specific count
            if filter_.event_types and len(filter_.event_types) == 1 and not (
                filter_.categories or filter_.min_priority or 
                filter_.start_time or filter_.end_time or filter_.custom_filter
            ):
                event_type = filter_.event_types[0]
                type_list_key = f"{self.key_prefix}:type:{event_type}"
                return self.redis.zcard(type_list_key)
            
            # For more complex filters, we need to get the events and count them
            # This is not efficient for large datasets, but works for now
            return len(self.get_events(filter_))
        
        except Exception as e:
            logger.error(f"Error getting event count from Redis: {e}")
            return 0
    
    def clear_events(self, filter_: Optional[EventFilter] = None) -> int:
        """
        Clear events from Redis.
        
        Args:
            filter_: Filter for events to clear
            
        Returns:
            Number of events cleared
        """
        try:
            # If no filter, clear all events
            if not filter_:
                # Get all event IDs
                event_ids = self.redis.zrange(self.event_list_key, 0, -1)
                count = len(event_ids)
                
                # Delete all event data
                for event_id in event_ids:
                    event_key = f"{self.key_prefix}:{event_id.decode()}"
                    self.redis.delete(event_key)
                
                # Get all event types
                event_types = self.redis.smembers(self.event_types_key)
                
                # Delete all type-specific lists
                for event_type in event_types:
                    type_list_key = f"{self.key_prefix}:type:{event_type.decode()}"
                    self.redis.delete(type_list_key)
                
                # Delete the main list and types set
                self.redis.delete(self.event_list_key)
                self.redis.delete(self.event_types_key)
                
                return count
            
            # For filtered clearing, get the events and delete them individually
            events = self.get_events(filter_)
            count = len(events)
            
            for event in events:
                # Remove from the main list
                self.redis.zrem(self.event_list_key, event.id)
                
                # Remove from the type list
                type_list_key = f"{self.key_prefix}:type:{event.event_type}"
                self.redis.zrem(type_list_key, event.id)
                
                # Remove the event data
                event_key = f"{self.key_prefix}:{event.id}"
                self.redis.delete(event_key)
            
            return count
        
        except Exception as e:
            logger.error(f"Error clearing events from Redis: {e}")
            return 0


class InMemoryEventPersistence(EventPersistence):
    """
    In-memory implementation of event persistence.
    
    This class stores events in memory for testing and development.
    """
    
    def __init__(self, max_events: int = 1000):
        """
        Initialize in-memory event persistence.
        
        Args:
            max_events: Maximum number of events to store
        """
        self.events: List[Event] = []
        self.max_events = max_events
    
    def store_event(self, event: Event) -> None:
        """
        Store an event in memory.
        
        Args:
            event: The event to store
        """
        self.events.append(event)
        
        # Trim the event list if it's too large
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def get_events(
        self,
        filter_: Optional[EventFilter] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Event]:
        """
        Get events from memory.
        
        Args:
            filter_: Filter for events
            limit: Maximum number of events to return
            offset: Offset for pagination
            
        Returns:
            List of events
        """
        # Apply filter
        if filter_:
            filtered_events = [e for e in self.events if filter_.matches(e)]
        else:
            filtered_events = self.events.copy()
        
        # Sort by timestamp
        filtered_events.sort(key=lambda e: e.timestamp)
        
        # Apply pagination
        paginated_events = filtered_events[offset:offset + limit if limit else None]
        
        return paginated_events
    
    def get_event_count(self, filter_: Optional[EventFilter] = None) -> int:
        """
        Get the count of events matching a filter.
        
        Args:
            filter_: Filter for events
            
        Returns:
            Count of matching events
        """
        if filter_:
            return len([e for e in self.events if filter_.matches(e)])
        else:
            return len(self.events)
    
    def clear_events(self, filter_: Optional[EventFilter] = None) -> int:
        """
        Clear events from memory.
        
        Args:
            filter_: Filter for events to clear
            
        Returns:
            Number of events cleared
        """
        if not filter_:
            count = len(self.events)
            self.events.clear()
            return count
        
        # Remove events that match the filter
        count = 0
        new_events = []
        for event in self.events:
            if filter_.matches(event):
                count += 1
            else:
                new_events.append(event)
        
        self.events = new_events
        return count


class BatchProcessor:
    """
    Processor for batched events.
    
    This class manages batches of events and processes them when ready.
    """
    
    def __init__(self):
        """Initialize the batch processor."""
        self.batches: Dict[str, EventBatch] = {}
        self.batch_handlers: Dict[str, Callable[[List[Event]], None]] = {}
        self.async_batch_handlers: Dict[str, Callable[[List[Event]], asyncio.coroutine]] = {}
    
    def register_batch_handler(
        self,
        event_type: str,
        handler: Callable[[List[Event]], None],
        max_size: int = 100,
        max_age_seconds: float = 5.0,
    ) -> None:
        """
        Register a batch handler for an event type.
        
        Args:
            event_type: The event type to handle
            handler: The handler function
            max_size: Maximum batch size
            max_age_seconds: Maximum batch age in seconds
        """
        self.batch_handlers[event_type] = handler
        self.batches[event_type] = EventBatch(
            event_type=event_type,
            max_size=max_size,
            max_age_seconds=max_age_seconds,
        )
    
    def register_async_batch_handler(
        self,
        event_type: str,
        handler: Callable[[List[Event]], asyncio.coroutine],
        max_size: int = 100,
        max_age_seconds: float = 5.0,
    ) -> None:
        """
        Register an async batch handler for an event type.
        
        Args:
            event_type: The event type to handle
            handler: The async handler function
            max_size: Maximum batch size
            max_age_seconds: Maximum batch age in seconds
        """
        self.async_batch_handlers[event_type] = handler
        self.batches[event_type] = EventBatch(
            event_type=event_type,
            max_size=max_size,
            max_age_seconds=max_age_seconds,
        )
    
    def add_event(self, event: Event) -> None:
        """
        Add an event to its batch.
        
        Args:
            event: The event to add
        """
        event_type = event.event_type
        
        # Check if we have a batch for this event type
        if event_type not in self.batches:
            return
        
        batch = self.batches[event_type]
        is_full = batch.add(event)
        
        # Process the batch if it's full
        if is_full:
            self._process_batch(event_type)
    
    def _process_batch(self, event_type: str) -> None:
        """
        Process a batch of events.
        
        Args:
            event_type: The event type to process
        """
        batch = self.batches[event_type]
        
        if not batch.events:
            return
        
        # Get a copy of the events
        events = batch.events.copy()
        
        # Clear the batch
        batch.clear()
        
        # Call the appropriate handler
        if event_type in self.batch_handlers:
            try:
                self.batch_handlers[event_type](events)
            except Exception as e:
                logger.error(f"Error in batch handler for {event_type}: {e}")
        
        elif event_type in self.async_batch_handlers:
            # Create a new event loop for the async handler
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.async_batch_handlers[event_type](events))
            except Exception as e:
                logger.error(f"Error in async batch handler for {event_type}: {e}")
            finally:
                loop.close()
    
    def process_ready_batches(self) -> None:
        """Process all batches that are ready."""
        for event_type, batch in self.batches.items():
            if batch.is_ready() and batch.events:
                self._process_batch(event_type)


class EventBus:
    """
    Central event bus for the application.
    
    This class manages event subscriptions, publishing, and persistence.
    """
    
    def __init__(self, persistence: Optional[EventPersistence] = None):
        """
        Initialize the event bus.
        
        Args:
            persistence: Event persistence implementation
        """
        # Map of event type to list of handlers
        self.handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        
        # Set of all registered event types
        self.registered_event_types: Set[str] = set()
        
        # Event persistence
        self.persistence = persistence
        
        # Batch processor
        self.batch_processor = BatchProcessor()
        
        # Async mode
        self.async_mode = False
        self.async_queue: Optional[asyncio.Queue] = None
        self.async_task: Optional[asyncio.Task] = None
    
    def register_handler(
        self,
        event_type: str,
        handler: Union[HandlerType, AsyncHandlerType],
        priority: EventPriority = EventPriority.NORMAL,
        filter_: Optional[EventFilter] = None,
    ) -> None:
        """
        Register an event handler.
        
        Args:
            event_type: The event type to handle
            handler: The handler function
            priority: The priority of this handler
            filter_: Additional filter for events
        """
        # Check if the handler is async
        is_async = asyncio.iscoroutinefunction(handler)
        
        # Create the handler wrapper
        handler_wrapper = EventHandler(
            handler=handler,
            event_type=event_type,
            priority=priority,
            is_async=is_async,
            filter_=filter_,
        )
        
        # Add to the handlers list
        self.handlers[event_type].append(handler_wrapper)
        
        # Sort handlers by priority (higher priority first)
        self.handlers[event_type].sort(key=lambda h: h.priority.value, reverse=True)
        
        # Add to registered event types
        self.registered_event_types.add(event_type)
        
        logger.debug(f"Registered handler for event type: {event_type}")
    
    def unregister_handler(self, event_type: str, handler: Callable) -> bool:
        """
        Unregister an event handler.
        
        Args:
            event_type: The event type
            handler: The handler function
            
        Returns:
            True if the handler was unregistered, False otherwise
        """
        if event_type not in self.handlers:
            return False
        
        # Find the handler wrapper
        for i, handler_wrapper in enumerate(self.handlers[event_type]):
            if handler_wrapper.handler == handler:
                # Remove the handler
                self.handlers[event_type].pop(i)
                
                # Clean up if there are no more handlers for this event type
                if not self.handlers[event_type]:
                    del self.handlers[event_type]
                    self.registered_event_types.remove(event_type)
                
                logger.debug(f"Unregistered handler for event type: {event_type}")
                return True
        
        return False
    
    def register_batch_handler(
        self,
        event_type: str,
        handler: Callable[[List[Event]], None],
        max_size: int = 100,
        max_age_seconds: float = 5.0,
    ) -> None:
        """
        Register a batch handler for an event type.
        
        Args:
            event_type: The event type to handle
            handler: The handler function
            max_size: Maximum batch size
            max_age_seconds: Maximum batch age in seconds
        """
        self.batch_processor.register_batch_handler(
            event_type=event_type,
            handler=handler,
            max_size=max_size,
            max_age_seconds=max_age_seconds,
        )
        
        # Add to registered event types
        self.registered_event_types.add(event_type)
        
        logger.debug(f"Registered batch handler for event type: {event_type}")
    
    def register_async_batch_handler(
        self,
        event_type: str,
        handler: Callable[[List[Event]], asyncio.coroutine],
        max_size: int = 100,
        max_age_seconds: float = 5.0,
    ) -> None:
        """
        Register an async batch handler for an event type.
        
        Args:
            event_type: The event type to handle
            handler: The async handler function
            max_size: Maximum batch size
            max_age_seconds: Maximum batch age in seconds
        """
        self.batch_processor.register_async_batch_handler(
            event_type=event_type,
            handler=handler,
            max_size=max_size,
            max_age_seconds=max_age_seconds,
        )
        
        # Add to registered event types
        self.registered_event_types.add(event_type)
        
        logger.debug(f"Registered async batch handler for event type: {event_type}")
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all registered handlers.
        
        Args:
            event: The event to publish
        """
        event_type = event.event_type
        
        # Store the event if persistence is enabled
        if self.persistence:
            try:
                self.persistence.store_event(event)
            except Exception as e:
                logger.error(f"Error storing event: {e}")
        
        # Add to batch processor if applicable
        self.batch_processor.add_event(event)
        
        # If in async mode, add to the queue
        if self.async_mode and self.async_queue:
            try:
                # Use asyncio.run_coroutine_threadsafe if called from a different thread
                if asyncio.get_event_loop().is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.async_queue.put(event),
                        asyncio.get_event_loop(),
                    )
                else:
                    # This will only work if called from the same thread as the event loop
                    self.async_queue.put_nowait(event)
                return
            except Exception as e:
                logger.error(f"Error adding event to async queue: {e}")
                # Fall back to sync mode
        
        # Get handlers for this event type
        handlers = self.handlers.get(event_type, [])
        
        # Call each handler
        for handler in handlers:
            if handler.matches(event):
                try:
                    handler.call(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
    
    async def publish_async(self, event: Event) -> None:
        """
        Publish an event asynchronously.
        
        Args:
            event: The event to publish
        """
        event_type = event.event_type
        
        # Store the event if persistence is enabled
        if self.persistence:
            try:
                # Run in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.persistence.store_event, event)
            except Exception as e:
                logger.error(f"Error storing event: {e}")
        
        # Add to batch processor if applicable
        await loop.run_in_executor(None, self.batch_processor.add_event, event)
        
        # Get handlers for this event type
        handlers = self.handlers.get(event_type, [])
        
        # Call each handler
        for handler in handlers:
            if handler.matches(event):
                try:
                    await handler.call_async(event)
                except Exception as e:
                    logger.error(f"Error in async event handler: {e}")
    
    async def _process_async_queue(self) -> None:
        """Process events from the async queue."""
        if not self.async_queue:
            return
        
        while True:
            try:
                # Get the next event from the queue
                event = await self.async_queue.get()
                
                # Process the event
                await self.publish_async(event)
                
                # Mark the task as done
                self.async_queue.task_done()
            except asyncio.CancelledError:
                # Task was cancelled, exit
                break
            except Exception as e:
                logger.error(f"Error processing event from async queue: {e}")
    
    def start_async_mode(self) -> None:
        """Start async mode for event processing."""
        if self.async_mode:
            return
        
        self.async_mode = True
        self.async_queue = asyncio.Queue()
        
        # Start the async task
        loop = asyncio.get_event_loop()
        self.async_task = loop.create_task(self._process_async_queue())
        
        logger.info("Started async event processing")
    
    def stop_async_mode(self) -> None:
        """Stop async mode for event processing."""
        if not self.async_mode:
            return
        
        self.async_mode = False
        
        # Cancel the async task
        if self.async_task:
            self.async_task.cancel()
            self.async_task = None
        
        self.async_queue = None
        
        logger.info("Stopped async event processing")
    
    def replay_events(
        self,
        filter_: Optional[EventFilter] = None,
        limit: Optional[int] = None,
    ) -> int:
        """
        Replay events from persistence.
        
        Args:
            filter_: Filter for events to replay
            limit: Maximum number of events to replay
            
        Returns:
            Number of events replayed
        """
        if not self.persistence:
            logger.warning("Cannot replay events: persistence not configured")
            return 0
        
        try:
            # Get events from persistence
            events = self.persistence.get_events(filter_=filter_, limit=limit)
            
            # Publish each event
            for event in events:
                self.publish(event)
            
            logger.info(f"Replayed {len(events)} events")
            return len(events)
        
        except Exception as e:
            logger.error(f"Error replaying events: {e}")
            return 0
    
    def process_batches(self) -> None:
        """Process all ready event batches."""
        self.batch_processor.process_ready_batches()


# Event class registry
_event_classes: Dict[str, Type[Event]] = {}


def _register_event_class(event_class: Type[Event]) -> None:
    """
    Register an event class in the registry.
    
    Args:
        event_class: The event class to register
    """
    event_type = getattr(event_class, "event_type", None)
    if event_type:
        _event_classes[event_type] = event_class


def _get_event_class_by_type(event_type: str) -> Optional[Type[Event]]:
    """
    Get an event class by its type.
    
    Args:
        event_type: The event type
        
    Returns:
        The event class, or None if not found
    """
    return _event_classes.get(event_type)


# Register all event classes defined in this module
def _register_all_event_classes() -> None:
    """Register all event classes defined in this module."""
    for name, obj in globals().items():
        if isinstance(obj, type) and issubclass(obj, Event) and obj != Event:
            _register_event_class(obj)


# Call this at module load time
_register_all_event_classes()


def subscribe(
    event_type: str,
    priority: EventPriority = EventPriority.NORMAL,
    filter_: Optional[EventFilter] = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator for subscribing to events.
    
    Args:
        event_type: The event type to subscribe to
        priority: The handler priority
        filter_: Additional filter for events
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Register the handler when the module is imported
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        
        # Store the subscription info on the function
        wrapper._event_subscription = {
            "event_type": event_type,
            "priority": priority,
            "filter": filter_,
        }
        
        return wrapper
    
    return decorator


def batch_subscribe(
    event_type: str,
    max_size: int = 100,
    max_age_seconds: float = 5.0,
) -> Callable[[Callable], Callable]:
    """
    Decorator for subscribing to batched events.
    
    Args:
        event_type: The event type to subscribe to
        max_size: Maximum batch size
        max_age_seconds: Maximum batch age in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Register the handler when the module is imported
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        
        # Store the batch subscription info on the function
        wrapper._batch_subscription = {
            "event_type": event_type,
            "max_size": max_size,
            "max_age_seconds": max_age_seconds,
        }
        
        return wrapper
    
    return decorator


def init_event_system(
    redis_client: Optional[redis.Redis] = None,
    max_events: int = 10000,
    async_mode: bool = False,
) -> EventBus:
    """
    Initialize the event system.
    
    Args:
        redis_client: Redis client for persistence (optional)
        max_events: Maximum number of events to store
        async_mode: Whether to use async mode
        
    Returns:
        The event bus instance
    """
    global _event_bus
    
    # Create persistence if Redis client is provided
    persistence = None
    if redis_client:
        try:
            persistence = RedisEventPersistence(
                redis_client=redis_client,
                max_events=max_events,
            )
            logger.info("Using Redis for event persistence")
        except Exception as e:
            logger.error(f"Error initializing Redis event persistence: {e}")
            # Fall back to in-memory persistence
            persistence = InMemoryEventPersistence(max_events=max_events)
            logger.info("Falling back to in-memory event persistence")
    else:
        # Use in-memory persistence
        persistence = InMemoryEventPersistence(max_events=max_events)
        logger.info("Using in-memory event persistence")
    
    # Create the event bus
    _event_bus = EventBus(persistence=persistence)
    
    # Start async mode if requested
    if async_mode:
        _event_bus.start_async_mode()
    
    # Find and register all handlers with @subscribe decorator
    _register_decorated_handlers()
    
    logger.info("Event system initialized")
    return _event_bus


def _register_decorated_handlers() -> None:
    """Register all handlers decorated with @subscribe."""
    # This is a simplified implementation that only works for handlers in the current module
    # For a real application, you would need to scan all modules for decorated handlers
    for name, obj in globals().items():
        if callable(obj) and hasattr(obj, "_event_subscription"):
            # Register the handler
            subscription = obj._event_subscription
            register_handler(
                event_type=subscription["event_type"],
                handler=obj,
                priority=subscription["priority"],
                filter_=subscription["filter"],
            )
        
        if callable(obj) and hasattr(obj, "_batch_subscription"):
            # Register the batch handler
            subscription = obj._batch_subscription
            register_batch_handler(
                event_type=subscription["event_type"],
                handler=obj,
                max_size=subscription["max_size"],
                max_age_seconds=subscription["max_age_seconds"],
            )


def get_event_bus() -> EventBus:
    """
    Get the global event bus instance.
    
    Returns:
        The event bus instance
        
    Raises:
        RuntimeError: If the event system has not been initialized
    """
    if _event_bus is None:
        raise RuntimeError("Event system not initialized. Call init_event_system() first.")
    
    return _event_bus


def register_handler(
    event_type: str,
    handler: Callable,
    priority: EventPriority = EventPriority.NORMAL,
    filter_: Optional[EventFilter] = None,
) -> None:
    """
    Register an event handler.
    
    Args:
        event_type: The event type to handle
        handler: The handler function
        priority: The priority of this handler
        filter_: Additional filter for events
        
    Raises:
        RuntimeError: If the event system has not been initialized
    """
    bus = get_event_bus()
    bus.register_handler(
        event_type=event_type,
        handler=handler,
        priority=priority,
        filter_=filter_,
    )


def unregister_handler(event_type: str, handler: Callable) -> bool:
    """
    Unregister an event handler.
    
    Args:
        event_type: The event type
        handler: The handler function
        
    Returns:
        True if the handler was unregistered, False otherwise
        
    Raises:
        RuntimeError: If the event system has not been initialized
    """
    bus = get_event_bus()
    return bus.unregister_handler(event_type=event_type, handler=handler)


def register_batch_handler(
    event_type: str,
    handler: Callable[[List[Event]], None],
    max_size: int = 100,
    max_age_seconds: float = 5.0,
) -> None:
    """
    Register a batch handler for an event type.
    
    Args:
        event_type: The event type to handle
        handler: The handler function
        max_size: Maximum batch size
        max_age_seconds: Maximum batch age in seconds
        
    Raises:
        RuntimeError: If the event system has not been initialized
    """
    bus = get_event_bus()
    bus.register_batch_handler(
        event_type=event_type,
        handler=handler,
        max_size=max_size,
        max_age_seconds=max_age_seconds,
    )


def publish_event(event_type: str, data: Dict[str, Any]) -> None:
    """
    Publish an event with the given type and data.
    
    This is a convenience function that creates an event object
    and publishes it to the event bus.
    
    Args:
        event_type: The event type
        data: The event data
        
    Raises:
        RuntimeError: If the event system has not been initialized
        ValueError: If the event type is not registered
    """
    # Get the event class for this type
    event_class = _get_event_class_by_type(event_type)
    if not event_class:
        # Try to create a generic event
        logger.warning(f"Unknown event type: {event_type}, using generic Event")
        event = Event(**data)
        event.event_type = event_type
    else:
        # Create the event object
        event = event_class(**data)
    
    # Publish the event
    bus = get_event_bus()
    bus.publish(event)


async def publish_event_async(event_type: str, data: Dict[str, Any]) -> None:
    """
    Publish an event asynchronously.
    
    Args:
        event_type: The event type
        data: The event data
        
    Raises:
        RuntimeError: If the event system has not been initialized
        ValueError: If the event type is not registered
    """
    # Get the event class for this type
    event_class = _get_event_class_by_type(event_type)
    if not event_class:
        # Try to create a generic event
        logger.warning(f"Unknown event type: {event_type}, using generic Event")
        event = Event(**data)
        event.event_type = event_type
    else:
        # Create the event object
        event = event_class(**data)
    
    # Publish the event
    bus = get_event_bus()
    await bus.publish_async(event)


def process_batches() -> None:
    """
    Process all ready event batches.
    
    Raises:
        RuntimeError: If the event system has not been initialized
    """
    bus = get_event_bus()
    bus.process_batches()


def replay_events(
    filter_: Optional[EventFilter] = None,
    limit: Optional[int] = None,
) -> int:
    """
    Replay events from persistence.
    
    Args:
        filter_: Filter for events to replay
        limit: Maximum number of events to replay
        
    Returns:
        Number of events replayed
        
    Raises:
        RuntimeError: If the event system has not been initialized
    """
    bus = get_event_bus()
    return bus.replay_events(filter_=filter_, limit=limit)
