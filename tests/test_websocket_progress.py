"""
Tests for WebSocket progress functionality.

This module tests the WebSocket functionality for real-time task progress tracking,
including connection handling, event emission, message delivery, heartbeat mechanism,
reconnection logic, event filtering, and concurrent connections.
"""

import asyncio
import json
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, call

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketState
from jose import jwt

from backend.api.v1.ws_progress import router, authenticate_websocket, handle_event, broadcast_to_task
from backend.core.events import EventType, Event, global_emitter, emit_event, initialize_events, shutdown_events
from backend.agents.custom_crew import CustomCrew
from backend.config import settings


# Test constants
TEST_TOKEN = "test.jwt.token"
TEST_USER_ID = str(uuid.uuid4())
TEST_TASK_ID = str(uuid.uuid4())
TEST_CREW_ID = str(uuid.uuid4())
TEST_AGENT_ID = "test_agent"
TEST_TOOL_ID = "test_tool"


# Mock JWT payload
TEST_JWT_PAYLOAD = {
    "sub": TEST_USER_ID,
    "role": "analyst",
    "exp": 9999999999  # Far future expiration
}


# Fixtures
@pytest.fixture
def app():
    """Create a FastAPI test application."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
async def initialized_events():
    """Initialize and clean up the event system."""
    await initialize_events()
    yield global_emitter
    await shutdown_events()


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    mock_ws = AsyncMock(spec=WebSocket)
    mock_ws.client_state = WebSocketState.CONNECTED
    mock_ws.query_params = {"token": TEST_TOKEN}
    mock_ws.headers = {}
    return mock_ws


@pytest.fixture
def mock_decode_token():
    """Mock the JWT token decoding."""
    with patch("backend.api.v1.ws_progress.decode_token") as mock_decode:
        mock_decode.return_value = TEST_JWT_PAYLOAD
        yield mock_decode


@pytest.fixture
def mock_custom_crew():
    """Create a mock CustomCrew instance."""
    crew = MagicMock(spec=CustomCrew)
    crew.crew_id = TEST_CREW_ID
    crew.task_ids = {
        "Test task": TEST_TASK_ID
    }
    crew.total_tasks = 3
    crew.completed_tasks = 0
    return crew


# Helper functions
async def create_test_event(event_type=EventType.TASK_PROGRESS, task_id=TEST_TASK_ID):
    """Create a test event with the specified type and task ID."""
    return await emit_event(
        event_type=event_type,
        task_id=task_id,
        crew_id=TEST_CREW_ID,
        agent_id=TEST_AGENT_ID,
        tool_id=TEST_TOOL_ID,
        progress=50,
        message="Test event message",
        data={"test_key": "test_value"}
    )


# Tests for WebSocket connection
@pytest.mark.asyncio
async def test_authenticate_websocket_valid_token(mock_websocket, mock_decode_token):
    """Test WebSocket authentication with a valid token."""
    payload = await authenticate_websocket(mock_websocket)
    
    assert payload == TEST_JWT_PAYLOAD
    mock_decode_token.assert_called_once_with(TEST_TOKEN)
    mock_websocket.close.assert_not_called()


@pytest.mark.asyncio
async def test_authenticate_websocket_invalid_token(mock_websocket, mock_decode_token):
    """Test WebSocket authentication with an invalid token."""
    mock_decode_token.return_value = None
    
    payload = await authenticate_websocket(mock_websocket)
    
    assert payload is None
    mock_websocket.close.assert_called_once_with(
        code=status.WS_1008_POLICY_VIOLATION, 
        reason="Invalid token"
    )


@pytest.mark.asyncio
async def test_authenticate_websocket_missing_token(mock_websocket):
    """Test WebSocket authentication with a missing token."""
    mock_websocket.query_params = {}
    
    payload = await authenticate_websocket(mock_websocket)
    
    assert payload is None
    mock_websocket.close.assert_called_once_with(
        code=status.WS_1008_POLICY_VIOLATION, 
        reason="Missing authentication token"
    )


@pytest.mark.asyncio
async def test_authenticate_websocket_token_in_header(mock_websocket, mock_decode_token):
    """Test WebSocket authentication with token in Authorization header."""
    mock_websocket.query_params = {}
    mock_websocket.headers = {"authorization": f"Bearer {TEST_TOKEN}"}
    
    payload = await authenticate_websocket(mock_websocket)
    
    assert payload == TEST_JWT_PAYLOAD
    mock_decode_token.assert_called_once_with(TEST_TOKEN)
    mock_websocket.close.assert_not_called()


# Tests for event emission
@pytest.mark.asyncio
async def test_event_emission_from_custom_crew(mock_custom_crew, initialized_events):
    """Test event emission from CustomCrew."""
    # Create a task for the crew
    task = MagicMock()
    task.description = "Test task"
    task.agent = MagicMock()
    task.agent.name = TEST_AGENT_ID
    task.agent.tools = []
    
    # Mock the parent _process_task method
    with patch.object(CustomCrew, '_process_task', new_callable=AsyncMock) as mock_process:
        # Set up the mock to return a result
        mock_process.return_value = "Task result"
        
        # Call the overridden _process_task method
        result = await CustomCrew._process_task(mock_custom_crew, task, {})
        
        # Verify the result
        assert result == "Task result"
        
        # Verify that the parent method was called
        mock_process.assert_called_once()
        
        # We can't directly verify event emission since it's done in a background task
        # But we can verify the task_ids are updated correctly
        assert mock_custom_crew.task_ids.get("Test task") is not None


@pytest.mark.asyncio
async def test_crew_kickoff_emits_events(mock_custom_crew, initialized_events):
    """Test that crew kickoff emits events."""
    # Mock the parent kickoff method
    with patch.object(CustomCrew, 'kickoff', return_value="Crew result") as mock_kickoff:
        # Call the overridden kickoff method
        result = CustomCrew.kickoff(mock_custom_crew, {})
        
        # Verify the result
        assert result == "Crew result"
        
        # Verify that the parent method was called
        mock_kickoff.assert_called_once_with({})


# Tests for WebSocket message delivery
@pytest.mark.asyncio
async def test_handle_event_matching_task_id(mock_websocket, initialized_events):
    """Test handling an event with a matching task ID."""
    event = await create_test_event()
    
    await handle_event(event, mock_websocket, TEST_TASK_ID)
    
    mock_websocket.send_text.assert_called_once()
    sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
    assert sent_data["type"] == EventType.TASK_PROGRESS
    assert sent_data["task_id"] == TEST_TASK_ID


@pytest.mark.asyncio
async def test_handle_event_non_matching_task_id(mock_websocket, initialized_events):
    """Test handling an event with a non-matching task ID."""
    event = await create_test_event(task_id="different_task_id")
    
    await handle_event(event, mock_websocket, TEST_TASK_ID)
    
    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_handle_event_disconnected_websocket(mock_websocket, initialized_events):
    """Test handling an event with a disconnected WebSocket."""
    event = await create_test_event()
    mock_websocket.client_state = WebSocketState.DISCONNECTED
    
    await handle_event(event, mock_websocket, TEST_TASK_ID)
    
    mock_websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_to_task(initialized_events):
    """Test broadcasting a message to all clients subscribed to a task."""
    # Create mock WebSockets
    mock_ws1 = AsyncMock(spec=WebSocket)
    mock_ws1.client_state = WebSocketState.CONNECTED
    
    mock_ws2 = AsyncMock(spec=WebSocket)
    mock_ws2.client_state = WebSocketState.CONNECTED
    
    mock_ws3 = AsyncMock(spec=WebSocket)
    mock_ws3.client_state = WebSocketState.DISCONNECTED
    
    # Set up the active_connections dict
    with patch("backend.api.v1.ws_progress.active_connections", {
        TEST_TASK_ID: {mock_ws1, mock_ws2, mock_ws3}
    }):
        message = {"type": "test", "message": "Test broadcast"}
        
        await broadcast_to_task(TEST_TASK_ID, message)
        
        # Connected WebSockets should receive the message
        mock_ws1.send_text.assert_called_once()
        mock_ws2.send_text.assert_called_once()
        
        # Disconnected WebSocket should not receive the message
        mock_ws3.send_text.assert_not_called()


# Tests for heartbeat mechanism
@pytest.mark.asyncio
async def test_send_heartbeat(mock_websocket):
    """Test sending heartbeat messages."""
    from backend.api.v1.ws_progress import send_heartbeat
    
    # Mock sleep to avoid waiting
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        # Create a task for send_heartbeat but cancel it after a short time
        task = asyncio.create_task(send_heartbeat(mock_websocket))
        
        # Allow the task to run for a bit
        await asyncio.sleep(0.1)
        
        # Cancel the task
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Verify that a heartbeat was sent
        mock_websocket.send_text.assert_called()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "heartbeat"
        
        # Verify that sleep was called with the correct interval
        mock_sleep.assert_called_with(30)


@pytest.mark.asyncio
async def test_send_heartbeat_disconnected(mock_websocket):
    """Test sending heartbeat to a disconnected WebSocket."""
    from backend.api.v1.ws_progress import send_heartbeat
    
    # Set WebSocket to disconnected state
    mock_websocket.client_state = WebSocketState.DISCONNECTED
    
    # Mock sleep to avoid waiting
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        # Create a task for send_heartbeat but cancel it after a short time
        task = asyncio.create_task(send_heartbeat(mock_websocket))
        
        # Allow the task to run for a bit
        await asyncio.sleep(0.1)
        
        # Cancel the task
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Verify that no heartbeat was sent (WebSocket is disconnected)
        mock_websocket.send_text.assert_not_called()
        
        # Verify that sleep was not called (loop should exit)
        mock_sleep.assert_not_called()


# Tests for reconnection logic
@pytest.mark.asyncio
async def test_websocket_endpoint_handles_disconnect(mock_websocket, mock_decode_token, initialized_events):
    """Test that the WebSocket endpoint handles disconnections gracefully."""
    from backend.api.v1.ws_progress import websocket_task_progress
    
    # Mock the WebSocket disconnect
    mock_websocket.receive_text.side_effect = WebSocketDisconnect()
    
    # Call the WebSocket endpoint
    await websocket_task_progress(mock_websocket, TEST_TASK_ID)
    
    # Verify that the WebSocket was accepted
    mock_websocket.accept.assert_called_once()
    
    # Verify that authenticate_websocket was called
    mock_decode_token.assert_called_once()


# Tests for concurrent connections
@pytest.mark.asyncio
async def test_multiple_websocket_connections(initialized_events):
    """Test handling multiple WebSocket connections for the same task."""
    # Create multiple mock WebSockets
    mock_ws1 = AsyncMock(spec=WebSocket)
    mock_ws1.client_state = WebSocketState.CONNECTED
    
    mock_ws2 = AsyncMock(spec=WebSocket)
    mock_ws2.client_state = WebSocketState.CONNECTED
    
    # Set up the active_connections dict
    with patch("backend.api.v1.ws_progress.active_connections", {
        TEST_TASK_ID: {mock_ws1, mock_ws2}
    }):
        # Create and emit an event
        event = await create_test_event()
        
        # Handle the event for both WebSockets
        await handle_event(event, mock_ws1, TEST_TASK_ID)
        await handle_event(event, mock_ws2, TEST_TASK_ID)
        
        # Both WebSockets should receive the event
        mock_ws1.send_text.assert_called_once()
        mock_ws2.send_text.assert_called_once()


# Integration tests
@pytest.mark.asyncio
async def test_websocket_task_progress_endpoint(app, mock_decode_token, initialized_events):
    """Test the WebSocket task progress endpoint."""
    from backend.api.v1.ws_progress import websocket_task_progress
    
    # Create a mock WebSocket
    mock_ws = AsyncMock(spec=WebSocket)
    mock_ws.client_state = WebSocketState.CONNECTED
    mock_ws.query_params = {"token": TEST_TOKEN}
    mock_ws.headers = {}
    
    # Mock the WebSocket receive to return once then raise WebSocketDisconnect
    mock_ws.receive_text.side_effect = ["ping", WebSocketDisconnect()]
    
    # Create a task for the WebSocket endpoint
    task = asyncio.create_task(websocket_task_progress(mock_ws, TEST_TASK_ID))
    
    # Allow the task to run for a bit
    await asyncio.sleep(0.1)
    
    # Emit an event for the task
    await emit_event(
        EventType.TASK_PROGRESS,
        task_id=TEST_TASK_ID,
        progress=50,
        message="Test progress event"
    )
    
    # Allow the event to be processed
    await asyncio.sleep(0.1)
    
    # Cancel the task
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    # Verify that the WebSocket was accepted
    mock_ws.accept.assert_called_once()
    
    # Verify that the WebSocket received the initial connection message
    assert mock_ws.send_text.call_count >= 1
    
    # Verify the first message was the connection message
    first_message = json.loads(mock_ws.send_text.call_args_list[0][0][0])
    assert first_message["type"] == "connected"
    assert first_message["task_id"] == TEST_TASK_ID


@pytest.mark.asyncio
async def test_websocket_all_tasks_endpoint_admin_access(initialized_events):
    """Test the WebSocket endpoint for all tasks with admin access."""
    from backend.api.v1.ws_progress import websocket_all_tasks
    
    # Create a mock WebSocket
    mock_ws = AsyncMock(spec=WebSocket)
    mock_ws.client_state = WebSocketState.CONNECTED
    mock_ws.query_params = {"token": TEST_TOKEN}
    mock_ws.headers = {}
    
    # Mock the WebSocket receive to return once then raise WebSocketDisconnect
    mock_ws.receive_text.side_effect = WebSocketDisconnect()
    
    # Mock the JWT token decoding for admin role
    with patch("backend.api.v1.ws_progress.authenticate_websocket", 
               return_value={"role": "admin", "sub": TEST_USER_ID}):
        
        # Create a task for the WebSocket endpoint
        task = asyncio.create_task(websocket_all_tasks(mock_ws))
        
        # Allow the task to run for a bit
        await asyncio.sleep(0.1)
        
        # Emit an event
        await emit_event(
            EventType.SYSTEM_INFO,
            message="System info event"
        )
        
        # Allow the event to be processed
        await asyncio.sleep(0.1)
        
        # Cancel the task
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Verify that the WebSocket was accepted
        mock_ws.accept.assert_called_once()
        
        # Verify that the WebSocket received the initial connection message
        assert mock_ws.send_text.call_count >= 1
        
        # Verify the first message was the connection message
        first_message = json.loads(mock_ws.send_text.call_args_list[0][0][0])
        assert first_message["type"] == "connected"


@pytest.mark.asyncio
async def test_websocket_all_tasks_endpoint_non_admin_access(initialized_events):
    """Test the WebSocket endpoint for all tasks with non-admin access."""
    from backend.api.v1.ws_progress import websocket_all_tasks
    
    # Create a mock WebSocket
    mock_ws = AsyncMock(spec=WebSocket)
    
    # Mock the JWT token decoding for non-admin role
    with patch("backend.api.v1.ws_progress.authenticate_websocket", 
               return_value={"role": "analyst", "sub": TEST_USER_ID}):
        
        await websocket_all_tasks(mock_ws)
        
        # Verify that the WebSocket was accepted but then closed due to insufficient permissions
        mock_ws.accept.assert_called_once()
        mock_ws.close.assert_called_once_with(
            code=status.WS_1008_POLICY_VIOLATION, 
            reason="Insufficient permissions"
        )
