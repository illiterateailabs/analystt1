import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useTaskProgress } from '../useTaskProgress';
import { useToast } from '../useToast';
import { useAuth } from '../useAuth';

// Mock WebSocket
class MockWebSocket {
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: (() => void) | null = null;
  readyState: number = MockWebSocket.CONNECTING;
  url: string;
  send = jest.fn();
  close = jest.fn(() => {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose();
    }
  });

  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor(url: string) {
    this.url = url;
    // Simulate async connection
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen();
      }
    }, 100);
  }

  // Helper to simulate receiving a message
  _simulateMessage(data: any) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
    }
  }

  // Helper to simulate an error
  _simulateError() {
    if (this.onerror) {
      this.onerror(new Event('error'));
    }
  }

  // Helper to simulate close
  _simulateClose() {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose();
    }
  }
}

Object.defineProperty(window, 'WebSocket', {
  writable: true,
  value: MockWebSocket,
});

// Mock useToast hook
jest.mock('../useToast', () => ({
  useToast: jest.fn(),
}));

// Mock useAuth hook
jest.mock('../useAuth', () => ({
  useAuth: jest.fn(),
}));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false, // Disable retries for tests
    },
  },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);

describe('useTaskProgress', () => {
  const mockToast = jest.fn();
  const mockUser = { id: 'user-123', username: 'testuser' };
  let wsInstance: MockWebSocket;

  beforeEach(() => {
    jest.clearAllMocks();
    queryClient.clear();
    (useToast as jest.Mock).mockReturnValue({ toast: mockToast });
    (useAuth as jest.Mock).mockReturnValue({ user: mockUser, isAuthenticated: true });

    // Ensure a fresh WebSocket instance for each test
    (window.WebSocket as jest.Mock) = jest.fn((url: string) => {
      wsInstance = new MockWebSocket(url);
      return wsInstance;
    });
  });

  afterEach(() => {
    jest.useRealTimers(); // Restore real timers after each test
  });

  // 1. WebSocket connection establishment with task ID
  test('should establish WebSocket connection with correct URL and task ID', async () => {
    const taskId = 'test-task-id';
    renderHook(() => useTaskProgress(taskId), { wrapper });

    expect(window.WebSocket).toHaveBeenCalledTimes(1);
    expect(window.WebSocket).toHaveBeenCalledWith(
      `ws://localhost:8000/api/v1/ws/progress/${taskId}?user_id=${mockUser.id}`
    );

    await waitFor(() => expect(wsInstance.readyState).toBe(MockWebSocket.OPEN));
    expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({
      description: 'Connected to task progress.',
      variant: 'success',
    }));
  });

  // 2. Receiving and processing progress events
  test('should receive and process progress events', async () => {
    const taskId = 'test-task-id';
    const { result } = renderHook(() => useTaskProgress(taskId), { wrapper });

    await waitFor(() => expect(result.current.isConnected).toBe(true));

    act(() => {
      wsInstance._simulateMessage({
        task_id: taskId,
        status: 'RUNNING',
        progress: 0.5,
        message: 'Processing data',
      });
    });

    await waitFor(() => {
      expect(result.current.progressData).toEqual({
        task_id: taskId,
        status: 'RUNNING',
        progress: 0.5,
        message: 'Processing data',
      });
    });

    act(() => {
      wsInstance._simulateMessage({
        task_id: taskId,
        status: 'COMPLETED',
        progress: 1.0,
        message: 'Task finished',
        result: { final: 'data' },
      });
    });

    await waitFor(() => {
      expect(result.current.progressData).toEqual({
        task_id: taskId,
        status: 'COMPLETED',
        progress: 1.0,
        message: 'Task finished',
        result: { final: 'data' },
      });
    });
  });

  // 3. Connection state management (connecting, connected, disconnected)
  test('should update connection state correctly', async () => {
    const taskId = 'test-task-id';
    const { result } = renderHook(() => useTaskProgress(taskId), { wrapper });

    expect(result.current.isConnected).toBe(false);
    expect(result.current.isConnecting).toBe(true);

    await waitFor(() => expect(result.current.isConnected).toBe(true));
    expect(result.current.isConnecting).toBe(false);

    act(() => {
      wsInstance._simulateClose();
    });

    await waitFor(() => expect(result.current.isConnected).toBe(false));
    expect(result.current.isConnecting).toBe(false);
  });

  // 4. Error handling for WebSocket failures
  test('should handle WebSocket errors and show toast', async () => {
    const taskId = 'test-task-id';
    renderHook(() => useTaskProgress(taskId), { wrapper });

    await waitFor(() => expect(wsInstance.readyState).toBe(MockWebSocket.OPEN));

    act(() => {
      wsInstance._simulateError();
    });

    await waitFor(() => expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({
      description: 'WebSocket error. Attempting to reconnect...',
      variant: 'destructive',
    })));
    expect(wsInstance.close).toHaveBeenCalledTimes(1); // Error should trigger close and reconnect attempt
  });

  // 5. Cleanup and disconnection on unmount
  test('should close WebSocket connection on unmount', async () => {
    const taskId = 'test-task-id';
    const { unmount } = renderHook(() => useTaskProgress(taskId), { wrapper });

    await waitFor(() => expect(wsInstance.readyState).toBe(MockWebSocket.OPEN));

    unmount();

    expect(wsInstance.close).toHaveBeenCalledTimes(1);
    expect(wsInstance.readyState).toBe(MockWebSocket.CLOSED);
  });

  // 6. Reconnection logic
  test('should attempt to reconnect on unexpected close', async () => {
    jest.useFakeTimers(); // Use fake timers for controlling setTimeout

    const taskId = 'test-task-id';
    renderHook(() => useTaskProgress(taskId), { wrapper });

    await waitFor(() => expect(wsInstance.readyState).toBe(MockWebSocket.OPEN));
    expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({
      description: 'Connected to task progress.',
      variant: 'success',
    }));
    mockToast.mockClear(); // Clear toast after initial connection

    // Simulate unexpected close (e.g., server restart)
    act(() => {
      wsInstance._simulateClose();
    });

    await waitFor(() => expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({
      description: 'WebSocket disconnected. Attempting to reconnect...',
      variant: 'destructive',
    })));

    // Advance timers to trigger reconnection attempt
    act(() => {
      jest.advanceTimersByTime(1000); // First reconnect attempt after 1 second
    });

    // A new WebSocket instance should be created
    expect(window.WebSocket).toHaveBeenCalledTimes(2);
    expect(window.WebSocket).toHaveBeenCalledWith(
      `ws://localhost:8000/api/v1/ws/progress/${taskId}?user_id=${mockUser.id}`
    );

    // Wait for the new connection to open
    await waitFor(() => expect(wsInstance.readyState).toBe(MockWebSocket.OPEN));
    expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({
      description: 'Reconnected to task progress.',
      variant: 'success',
    }));
  });

  // 7. Progress data updates and state management
  test('should reset progress data when task ID changes', async () => {
    const taskId1 = 'task-id-1';
    const taskId2 = 'task-id-2';
    const { result, rerender } = renderHook(({ taskId }) => useTaskProgress(taskId), {
      wrapper,
      initialProps: { taskId: taskId1 },
    });

    await waitFor(() => expect(result.current.isConnected).toBe(true));

    act(() => {
      wsInstance._simulateMessage({
        task_id: taskId1,
        status: 'RUNNING',
        progress: 0.5,
        message: 'Processing data for task 1',
      });
    });

    await waitFor(() => {
      expect(result.current.progressData?.task_id).toBe(taskId1);
    });

    // Rerender with new task ID
    rerender({ taskId: taskId2 });

    // Expect progressData to be reset
    expect(result.current.progressData).toBeNull();
    expect(wsInstance.close).toHaveBeenCalledTimes(1); // Old connection closed

    // New connection should be established for taskId2
    await waitFor(() => expect(result.current.isConnected).toBe(true));
    expect(window.WebSocket).toHaveBeenCalledTimes(2);
    expect(window.WebSocket).toHaveBeenCalledWith(
      `ws://localhost:8000/api/v1/ws/progress/${taskId2}?user_id=${mockUser.id}`
    );

    act(() => {
      wsInstance._simulateMessage({
        task_id: taskId2,
        status: 'RUNNING',
        progress: 0.8,
        message: 'Processing data for task 2',
      });
    });

    await waitFor(() => {
      expect(result.current.progressData).toEqual({
        task_id: taskId2,
        status: 'RUNNING',
        progress: 0.8,
        message: 'Processing data for task 2',
      });
    });
  });

  // 8. Event filtering by task ID
  test('should only process messages for the current task ID', async () => {
    const taskId = 'current-task';
    const otherTaskId = 'other-task';
    const { result } = renderHook(() => useTaskProgress(taskId), { wrapper });

    await waitFor(() => expect(result.current.isConnected).toBe(true));

    act(() => {
      // Message for a different task ID should be ignored
      wsInstance._simulateMessage({
        task_id: otherTaskId,
        status: 'RUNNING',
        progress: 0.1,
        message: 'Message for other task',
      });
    });

    // Progress data should remain null or initial state
    expect(result.current.progressData).toBeNull();

    act(() => {
      // Message for the current task ID should be processed
      wsInstance._simulateMessage({
        task_id: taskId,
        status: 'RUNNING',
        progress: 0.5,
        message: 'Message for current task',
      });
    });

    await waitFor(() => {
      expect(result.current.progressData).toEqual({
        task_id: taskId,
        status: 'RUNNING',
        progress: 0.5,
        message: 'Message for current task',
      });
    });
  });

  test('should not connect if user is not authenticated', async () => {
    (useAuth as jest.Mock).mockReturnValue({ user: null, isAuthenticated: false });
    const taskId = 'test-task-id';
    renderHook(() => useTaskProgress(taskId), { wrapper });

    expect(window.WebSocket).not.toHaveBeenCalled();
    expect(mockToast).not.toHaveBeenCalled();
  });

  test('should not connect if taskId is null or undefined', async () => {
    renderHook(() => useTaskProgress(null), { wrapper });
    expect(window.WebSocket).not.toHaveBeenCalled();

    renderHook(() => useTaskProgress(undefined), { wrapper });
    expect(window.WebSocket).not.toHaveBeenCalled();
  });
});
