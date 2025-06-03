import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Typography,
  LinearProgress,
  Paper,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Collapse,
  IconButton,
  Divider,
  Alert,
  CircularProgress,
  useTheme,
  styled
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  Code as CodeIcon,
  Person as PersonIcon,
  Group as GroupIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Refresh as RefreshIcon,
  Warning as WarningIcon,
  Storage as StorageIcon,
  Timeline as TimelineIcon,
  SignalWifi4Bar as ConnectedIcon,
  SignalWifiOff as DisconnectedIcon,
  SignalWifiStatusbarConnectedNoInternet4 as ReconnectingIcon
} from '@mui/icons-material';
import { useTaskProgress, EventType, ConnectionState, TaskStatus, TaskEvent } from '../../hooks/useTaskProgress';

// Styled components
const EventListItem = styled(ListItem)(({ theme }) => ({
  marginBottom: theme.spacing(1),
  borderRadius: theme.shape.borderRadius,
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  }
}));

const TimelineConnector = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 0,
  bottom: 0,
  left: 20,
  width: 2,
  backgroundColor: theme.palette.divider,
  zIndex: 0
}));

// Props interface
interface TaskProgressProps {
  taskId: string;
  token: string;
  baseUrl?: string;
  onComplete?: (events: TaskEvent[]) => void;
  showTimeline?: boolean;
  maxEvents?: number;
}

// Helper function to format timestamp
const formatTime = (timestamp: string): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

// Helper function to truncate text
const truncateText = (text: string, maxLength: number = 100): string => {
  if (!text) return '';
  return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
};

// Component for task progress
const TaskProgress: React.FC<TaskProgressProps> = ({
  taskId,
  token,
  baseUrl,
  onComplete,
  showTimeline = true,
  maxEvents = 50
}) => {
  const theme = useTheme();
  const [expandedEvents, setExpandedEvents] = useState<Record<string, boolean>>({});
  const [displayedEvents, setDisplayedEvents] = useState<TaskEvent[]>([]);

  // Use the task progress hook
  const {
    events,
    progress,
    status,
    connectionState,
    latestEvent,
    error,
    reconnect,
    clearEvents
  } = useTaskProgress(taskId, token, baseUrl);

  // Toggle event expansion
  const toggleEventExpand = (eventId: string) => {
    setExpandedEvents(prev => ({
      ...prev,
      [eventId]: !prev[eventId]
    }));
  };

  // Filter and limit events for display
  useEffect(() => {
    // Filter out heartbeat events and limit to maxEvents
    const filteredEvents = events
      .filter(event => event.type !== EventType.HEARTBEAT && event.type !== EventType.PONG)
      .slice(-maxEvents);
    
    setDisplayedEvents(filteredEvents);
  }, [events, maxEvents]);

  // Call onComplete callback when task is completed
  useEffect(() => {
    if (status === TaskStatus.COMPLETED && onComplete) {
      onComplete(events);
    }
  }, [status, events, onComplete]);

  // Get icon and color for event type
  const getEventIcon = (eventType: EventType): { icon: React.ReactNode; color: string } => {
    switch (eventType) {
      case EventType.CREW_STARTED:
      case EventType.TASK_STARTED:
        return { icon: <PlayIcon />, color: theme.palette.info.main };
      
      case EventType.CREW_COMPLETED:
      case EventType.TASK_COMPLETED:
      case EventType.AGENT_COMPLETED:
        return { icon: <CheckIcon />, color: theme.palette.success.main };
      
      case EventType.CREW_FAILED:
      case EventType.TASK_FAILED:
      case EventType.AGENT_FAILED:
      case EventType.TOOL_FAILED:
        return { icon: <ErrorIcon />, color: theme.palette.error.main };
      
      case EventType.AGENT_STARTED:
      case EventType.AGENT_PROGRESS:
        return { icon: <PersonIcon />, color: theme.palette.primary.main };
      
      case EventType.TOOL_STARTED:
      case EventType.TOOL_COMPLETED:
        return { icon: <CodeIcon />, color: theme.palette.secondary.main };
      
      case EventType.HITL_REVIEW_REQUESTED:
      case EventType.HITL_REVIEW_APPROVED:
      case EventType.HITL_REVIEW_REJECTED:
        return { icon: <PersonIcon />, color: theme.palette.warning.main };
      
      case EventType.SYSTEM_INFO:
        return { icon: <TimelineIcon />, color: theme.palette.info.main };
      
      case EventType.SYSTEM_WARNING:
        return { icon: <WarningIcon />, color: theme.palette.warning.main };
      
      case EventType.SYSTEM_ERROR:
        return { icon: <ErrorIcon />, color: theme.palette.error.main };
      
      default:
        return { icon: <StorageIcon />, color: theme.palette.text.secondary };
    }
  };

  // Get status chip for current task status
  const getStatusChip = useMemo(() => {
    switch (status) {
      case TaskStatus.PENDING:
        return <Chip 
          label="Pending" 
          color="default" 
          size="small" 
          icon={<TimelineIcon />} 
        />;
      case TaskStatus.RUNNING:
        return <Chip 
          label="Running" 
          color="primary" 
          size="small" 
          icon={<PlayIcon />} 
        />;
      case TaskStatus.PAUSED:
        return <Chip 
          label="Paused" 
          color="warning" 
          size="small" 
          icon={<WarningIcon />} 
        />;
      case TaskStatus.COMPLETED:
        return <Chip 
          label="Completed" 
          color="success" 
          size="small" 
          icon={<CheckIcon />} 
        />;
      case TaskStatus.FAILED:
        return <Chip 
          label="Failed" 
          color="error" 
          size="small" 
          icon={<ErrorIcon />} 
        />;
      default:
        return <Chip 
          label="Unknown" 
          color="default" 
          size="small" 
        />;
    }
  }, [status]);

  // Get connection status indicator
  const getConnectionIndicator = useMemo(() => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return <Chip 
          label="Connected" 
          color="success" 
          size="small" 
          icon={<ConnectedIcon />} 
          sx={{ ml: 1 }}
        />;
      case ConnectionState.CONNECTING:
        return <Chip 
          label="Connecting" 
          color="info" 
          size="small" 
          icon={<CircularProgress size={16} />} 
          sx={{ ml: 1 }}
        />;
      case ConnectionState.RECONNECTING:
        return <Chip 
          label="Reconnecting" 
          color="warning" 
          size="small" 
          icon={<ReconnectingIcon />} 
          sx={{ ml: 1 }}
          onClick={reconnect}
        />;
      case ConnectionState.DISCONNECTED:
        return <Chip 
          label="Disconnected" 
          color="error" 
          size="small" 
          icon={<DisconnectedIcon />} 
          sx={{ ml: 1 }}
          onClick={reconnect}
        />;
      case ConnectionState.ERROR:
        return <Chip 
          label="Error" 
          color="error" 
          size="small" 
          icon={<ErrorIcon />} 
          sx={{ ml: 1 }}
          onClick={reconnect}
        />;
      default:
        return null;
    }
  }, [connectionState, reconnect]);

  // Render loading state
  if (!taskId) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" p={3}>
        <Typography variant="body1" color="textSecondary">
          No task selected
        </Typography>
      </Box>
    );
  }

  return (
    <Paper elevation={2} sx={{ p: 2, mb: 3 }}>
      {/* Header with status and connection indicator */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Box display="flex" alignItems="center">
          <Typography variant="h6" component="h2">
            Task Progress
          </Typography>
          {getStatusChip}
          {getConnectionIndicator}
        </Box>
        <Box>
          <IconButton 
            size="small" 
            onClick={() => clearEvents()} 
            title="Clear Events"
            color="primary"
          >
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Progress bar */}
      <Box mb={2}>
        <Box display="flex" justifyContent="space-between" mb={0.5}>
          <Typography variant="body2" color="textSecondary">
            {progress < 100 ? 'In Progress' : 'Complete'}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            {Math.round(progress)}%
          </Typography>
        </Box>
        <LinearProgress 
          variant="determinate" 
          value={progress} 
          color={status === TaskStatus.FAILED ? "error" : "primary"}
          sx={{ height: 8, borderRadius: 4 }}
        />
      </Box>

      {/* Latest event message */}
      {latestEvent && latestEvent.message && (
        <Box mb={2}>
          <Typography variant="body2" fontWeight="medium">
            {latestEvent.message}
          </Typography>
        </Box>
      )}

      {/* Error display */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error.message}
          <IconButton 
            size="small" 
            onClick={reconnect} 
            sx={{ ml: 1 }}
            title="Reconnect"
          >
            <RefreshIcon fontSize="small" />
          </IconButton>
        </Alert>
      )}

      {/* Timeline of events */}
      {showTimeline && (
        <>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle2" gutterBottom>
            Event Timeline
          </Typography>
          
          {displayedEvents.length === 0 ? (
            <Box display="flex" justifyContent="center" alignItems="center" p={2}>
              <Typography variant="body2" color="textSecondary">
                No events yet
              </Typography>
            </Box>
          ) : (
            <Box position="relative">
              <TimelineConnector />
              <List disablePadding>
                {displayedEvents.map((event, index) => {
                  const { icon, color } = getEventIcon(event.type as EventType);
                  const isExpanded = expandedEvents[event.id] || false;
                  
                  return (
                    <EventListItem 
                      key={event.id} 
                      disablePadding 
                      sx={{ pl: 0, zIndex: 1 }}
                      secondaryAction={
                        <IconButton 
                          edge="end" 
                          size="small" 
                          onClick={() => toggleEventExpand(event.id)}
                        >
                          {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                        </IconButton>
                      }
                    >
                      <Box width="100%">
                        <Box display="flex" alignItems="flex-start">
                          <ListItemIcon sx={{ minWidth: 40, color }}>
                            {icon}
                          </ListItemIcon>
                          <ListItemText
                            primary={
                              <Box display="flex" justifyContent="space-between">
                                <Typography variant="body2" fontWeight="medium">
                                  {event.type.split('_').map(word => 
                                    word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
                                  ).join(' ')}
                                </Typography>
                                <Typography variant="caption" color="textSecondary">
                                  {formatTime(event.timestamp_iso)}
                                </Typography>
                              </Box>
                            }
                            secondary={
                              <Typography variant="body2" color="textSecondary">
                                {truncateText(event.message || '', 60)}
                              </Typography>
                            }
                          />
                        </Box>
                        
                        <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                          <Box pl={5} pr={2} py={1} bgcolor={theme.palette.action.hover} borderRadius={1} mt={1}>
                            {event.agent_id && (
                              <Typography variant="body2" gutterBottom>
                                <strong>Agent:</strong> {event.agent_id}
                              </Typography>
                            )}
                            {event.tool_id && (
                              <Typography variant="body2" gutterBottom>
                                <strong>Tool:</strong> {event.tool_id}
                              </Typography>
                            )}
                            {event.message && (
                              <Typography variant="body2" gutterBottom>
                                <strong>Message:</strong> {event.message}
                              </Typography>
                            )}
                            {event.data && Object.keys(event.data).length > 0 && (
                              <Box mt={1}>
                                <Typography variant="body2" fontWeight="medium" gutterBottom>
                                  Data:
                                </Typography>
                                <Box 
                                  component="pre" 
                                  sx={{ 
                                    p: 1, 
                                    bgcolor: 'background.paper', 
                                    borderRadius: 1,
                                    fontSize: '0.75rem',
                                    overflow: 'auto',
                                    maxHeight: 200
                                  }}
                                >
                                  {JSON.stringify(event.data, null, 2)}
                                </Box>
                              </Box>
                            )}
                          </Box>
                        </Collapse>
                      </Box>
                    </EventListItem>
                  );
                })}
              </List>
            </Box>
          )}
        </>
      )}
    </Paper>
  );
};

export default TaskProgress;
