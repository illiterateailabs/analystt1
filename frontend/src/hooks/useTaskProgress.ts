import { useEffect, useState, useCallback, useRef } from 'react';

// Event types from backend
export enum EventType {
  // Task lifecycle events
  TASK_STARTED = "task_started",
  TASK_PROGRESS = "task_progress",
  TASK_COMPLETED = "task_completed",
  TASK_FAILED = "task_failed",
  
  // Agent lifecycle events
  AGENT_STARTED = "agent_started",
  AGENT_PROGRESS = "agent_progress",
  AGENT_COMPLETED = "agent_completed",
  AGENT_FAILED = "agent_failed",
  
  // Tool events
  TOOL_STARTED = "tool_started",
  TOOL_COMPLETED = "tool_completed",
  TOOL_FAILED = "tool_failed",
  
  // Crew lifecycle events
  CREW_STARTED = "crew_started",
  CREW_PROGRESS = "crew_progress",
  CREW_COMPLETED = "crew_completed",
  CREW_FAILED = "crew_failed",
  
  // HITL events
  HITL_REVIEW_REQUESTED = "hitl_review_requested",
  HITL_REVIEW_APPROVED = "hitl_review_approved",
  HITL_REVIEW_REJECTED = "hitl_review_rejected",
  
  // System events
  SYSTEM_INFO = "system_info",
  SYSTEM_WARNING = "system_warning",
  SYSTEM_ERROR = "system_error",
  
  // WebSocket events
  CONNECTED = "connected",
  HEARTBEAT = "heartbeat",
  PONG = "pong"
}

// Connection states
export enum ConnectionState {
  CONNECTING = "connecting",
  CONNECTED = "connected",
  DISCONNECTED = "disconnected",
  RECONNECTING = "reconnecting",
  ERROR = "error"
}

// Task status
export enum TaskStatus {
  PENDING = "pending",
  RUNNING = "running",
  PAUSED = "paused",
  COMPLETED = "completed",
  FAILED = "failed"
}

// Event interface
export interface TaskEvent {
  id: string;
  type: EventType;
  timestamp: number;
  timestamp_iso: string;
  task_id?: string;
  crew_id?: string;
  agent_id?: string;
  tool_id?: string;
  progress?: number;
  status?: string;
  message?: string;
  data?: Record<string, any>;
}

// Event handler type
export type EventHandler = (event: TaskEvent) => void;

// WebSocket options
interface WebSocketOptions {
  reconnectAttempts?: number;
  reconnectInterval?: number;
  maxReconnectInterval?: number;
  reconnectDecay?: number;
  heartbeatInterval?: number;
  autoReconnect?: boolean;
}

// Hook return type
interface UseTaskProgressReturn {
  events: TaskEvent[];
  progress: number;
  status: TaskStatus;
  connectionState: ConnectionState;
  latestEvent: TaskEvent | null;
  error: Error | null;
  subscribe: (eventType: EventType | null, handler: EventHandler) => () => void;
  clearEvents: () => void;
  reconnect: () => void;
  getEventsByType: (eventType: EventType) => TaskEvent[];
  getLatestEventByType: (eventType: EventType) => TaskEvent | null;
}

// Default options
const defaultOptions: WebSocketOptions = {
  reconnectAttempts: 5,
  reconnectInterval: 1000, // Start with 1 second
  maxReconnectInterval: 30000, // Max 30 seconds
  reconnectDecay: 1.5, // Exponential backoff factor
  heartbeatInterval: 30000, // 30 seconds
  autoReconnect: true
};

/**
 * Hook for WebSocket task progress tracking
 * 
 * @param taskId The ID of the task to track
 * @param token JWT authentication token
 * @param baseUrl Base URL for the WebSocket connection
 * @param options WebSocket connection options
 * @returns Progress tracking state and methods
 */
export function useTaskProgress(
  taskId: string | null,
  token: string,
  baseUrl: string = window.location.origin.replace('http', 'ws'),
  options: WebSocketOptions = {}
): UseTaskProgressReturn {
  // Merge options with defaults
  const wsOptions = { ...defaultOptions, ...options };
  
  // State
  const [events, setEvents] = useState<TaskEvent[]>([]);
  const [progress, setProgress] = useState<number>(0);
  const [status, setStatus] = useState<TaskStatus>(TaskStatus.PENDING);
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const [latestEvent, setLatestEvent] = useState<TaskEvent | null>(null);
  const [error, setError] = useState<Error | null>(null);
  
  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef<number>(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const eventHandlersRef = useRef<Map<EventType | null, Set<EventHandler>>>(new Map());
  
  // Calculate WebSocket URL
  const getWebSocketUrl = useCallback(() => {
    if (!taskId) return null;
    
    const wsUrl = new URL(`${baseUrl}/api/v1/ws/tasks/${taskId}`);
    wsUrl.searchParams.append('token', token);
    return wsUrl.toString();
  }, [baseUrl, taskId, token]);
  
  // Handle incoming WebSocket messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data) as TaskEvent;
      
      // Update latest event
      setLatestEvent(data);
      
      // Add to event history
      setEvents(prev => [...prev, data]);
      
      // Update progress if available
      if (data.progress !== undefined) {
        setProgress(data.progress);
      }
      
      // Update status based on event type
      if (data.type === EventType.TASK_COMPLETED || data.type === EventType.CREW_COMPLETED) {
        setStatus(TaskStatus.COMPLETED);
      } else if (data.type === EventType.TASK_FAILED || data.type === EventType.CREW_FAILED) {
        setStatus(TaskStatus.FAILED);
      } else if (data.type === EventType.TASK_STARTED || data.type === EventType.CREW_STARTED) {
        setStatus(TaskStatus.RUNNING);
      } else if (data.type === EventType.HITL_REVIEW_REQUESTED) {
        setStatus(TaskStatus.PAUSED);
      }
      
      // Notify event handlers
      const handlers = eventHandlersRef.current.get(data.type) || new Set();
      const globalHandlers = eventHandlersRef.current.get(null) || new Set();
      
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (err) {
          console.error('Error in event handler:', err);
        }
      });
      
      globalHandlers.forEach(handler => {
        try {
          handler(data);
        } catch (err) {
          console.error('Error in global event handler:', err);
        }
      });
      
      // Respond to heartbeat
      if (data.type === EventType.HEARTBEAT) {
        wsRef.current?.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
      }
    } catch (err) {
      console.error('Error parsing WebSocket message:', err);
    }
  }, []);
  
  // Connect to WebSocket
  const connect = useCallback(() => {
    // Don't connect if taskId or token is missing
    if (!taskId || !token) {
      return;
    }
    
    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    // Clear existing timeouts
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    // Update connection state
    setConnectionState(ConnectionState.CONNECTING);
    
    // Create new WebSocket connection
    const wsUrl = getWebSocketUrl();
    if (!wsUrl) return;
    
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    
    // Setup event handlers
    ws.onopen = () => {
      console.log(`WebSocket connected to ${wsUrl}`);
      setConnectionState(ConnectionState.CONNECTED);
      reconnectAttemptRef.current = 0;
      
      // Setup heartbeat interval
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
      }
      
      heartbeatIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
        }
      }, wsOptions.heartbeatInterval);
    };
    
    ws.onmessage = handleMessage;
    
    ws.onclose = (event) => {
      console.log(`WebSocket disconnected: ${event.code} ${event.reason}`);
      setConnectionState(ConnectionState.DISCONNECTED);
      
      // Clear heartbeat interval
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
        heartbeatIntervalRef.current = null;
      }
      
      // Attempt to reconnect if enabled
      if (wsOptions.autoReconnect) {
        reconnect();
      }
    };
    
    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      setConnectionState(ConnectionState.ERROR);
      setError(new Error('WebSocket connection error'));
    };
  }, [taskId, token, getWebSocketUrl, handleMessage, wsOptions.autoReconnect, wsOptions.heartbeatInterval]);
  
  // Reconnect to WebSocket with exponential backoff
  const reconnect = useCallback(() => {
    if (reconnectAttemptRef.current >= (wsOptions.reconnectAttempts || 0)) {
      console.log('Maximum reconnection attempts reached');
      return;
    }
    
    // Calculate backoff delay
    const delay = Math.min(
      wsOptions.reconnectInterval! * Math.pow(wsOptions.reconnectDecay!, reconnectAttemptRef.current),
      wsOptions.maxReconnectInterval!
    );
    
    console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttemptRef.current + 1})`);
    setConnectionState(ConnectionState.RECONNECTING);
    
    // Schedule reconnection
    reconnectTimeoutRef.current = setTimeout(() => {
      reconnectAttemptRef.current++;
      connect();
    }, delay);
  }, [connect, wsOptions.reconnectAttempts, wsOptions.reconnectInterval, wsOptions.reconnectDecay, wsOptions.maxReconnectInterval]);
  
  // Subscribe to events
  const subscribe = useCallback((eventType: EventType | null, handler: EventHandler) => {
    if (!eventHandlersRef.current.has(eventType)) {
      eventHandlersRef.current.set(eventType, new Set());
    }
    
    const handlers = eventHandlersRef.current.get(eventType)!;
    handlers.add(handler);
    
    // Return unsubscribe function
    return () => {
      const handlers = eventHandlersRef.current.get(eventType);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          eventHandlersRef.current.delete(eventType);
        }
      }
    };
  }, []);
  
  // Clear events
  const clearEvents = useCallback(() => {
    setEvents([]);
    setLatestEvent(null);
  }, []);
  
  // Get events by type
  const getEventsByType = useCallback((eventType: EventType) => {
    return events.filter(event => event.type === eventType);
  }, [events]);
  
  // Get latest event by type
  const getLatestEventByType = useCallback((eventType: EventType) => {
    const filteredEvents = events.filter(event => event.type === eventType);
    return filteredEvents.length > 0 ? filteredEvents[filteredEvents.length - 1] : null;
  }, [events]);
  
  // Connect on mount and when taskId or token changes
  useEffect(() => {
    if (taskId && token) {
      connect();
    }
    
    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
        heartbeatIntervalRef.current = null;
      }
    };
  }, [taskId, token, connect]);
  
  return {
    events,
    progress,
    status,
    connectionState,
    latestEvent,
    error,
    subscribe,
    clearEvents,
    reconnect,
    getEventsByType,
    getLatestEventByType
  };
}
