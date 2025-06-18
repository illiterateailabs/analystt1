import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import TaskProgress from '../TaskProgress';
import { useTaskProgress } from '../../../hooks/useTaskProgress';

// Mock the useTaskProgress hook
jest.mock('../../../hooks/useTaskProgress', () => ({
  useTaskProgress: jest.fn(),
}));

describe('TaskProgress', () => {
  const mockUseTaskProgress = useTaskProgress as jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    // Default mock implementation for useTaskProgress
    mockUseTaskProgress.mockReturnValue({
      progressData: null,
      isConnected: false,
      isConnecting: false,
    });
  });

  // 1. Rendering with initial progress data (status, message, percentage).
  test('renders with initial progress data (running)', () => {
    mockUseTaskProgress.mockReturnValue({
      progressData: {
        task_id: 'task-123',
        status: 'RUNNING',
        progress: 0.25,
        message: 'Fetching initial data...',
        start_time: Date.now() / 1000 - 60, // 60 seconds ago
      },
      isConnected: true,
      isConnecting: false,
    });

    render(<TaskProgress taskId="task-123" />);

    expect(screen.getByText('Status: RUNNING')).toBeInTheDocument();
    expect(screen.getByText('Fetching initial data...')).toBeInTheDocument();
    expect(screen.getByText('25%')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '25');
    expect(screen.getByText(/Time Elapsed:/)).toBeInTheDocument();
  });

  test('renders with initial progress data (pending)', () => {
    mockUseTaskProgress.mockReturnValue({
      progressData: {
        task_id: 'task-123',
        status: 'PENDING',
        progress: 0,
        message: 'Task queued...',
        start_time: Date.now() / 1000,
      },
      isConnected: true,
      isConnecting: false,
    });

    render(<TaskProgress taskId="task-123" />);

    expect(screen.getByText('Status: PENDING')).toBeInTheDocument();
    expect(screen.getByText('Task queued...')).toBeInTheDocument();
    expect(screen.getByText('0%')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '0');
  });

  // 2. Updating progress display when progressData prop changes.
  test('updates progress display when progressData changes', async () => {
    let currentProgressData = {
      task_id: 'task-123',
      status: 'RUNNING',
      progress: 0.1,
      message: 'Step 1 of 3',
      start_time: Date.now() / 1000 - 10,
    };

    mockUseTaskProgress.mockReturnValue({
      progressData: currentProgressData,
      isConnected: true,
      isConnecting: false,
    });

    const { rerender } = render(<TaskProgress taskId="task-123" />);

    expect(screen.getByText('10%')).toBeInTheDocument();
    expect(screen.getByText('Step 1 of 3')).toBeInTheDocument();

    // Simulate progress update
    currentProgressData = {
      ...currentProgressData,
      progress: 0.75,
      message: 'Step 2 of 3: Processing results',
      start_time: Date.now() / 1000 - 30,
    };

    mockUseTaskProgress.mockReturnValue({
      progressData: currentProgressData,
      isConnected: true,
      isConnecting: false,
    });

    rerender(<TaskProgress taskId="task-123" />);

    await waitFor(() => {
      expect(screen.getByText('75%')).toBeInTheDocument();
      expect(screen.getByText('Step 2 of 3: Processing results')).toBeInTheDocument();
      expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '75');
    });
  });

  // 3. Handling different task statuses (pending, running, completed, failed).
  test('displays completed status correctly', () => {
    mockUseTaskProgress.mockReturnValue({
      progressData: {
        task_id: 'task-123',
        status: 'COMPLETED',
        progress: 1.0,
        message: 'Task finished successfully!',
        start_time: Date.now() / 1000 - 120,
      },
      isConnected: true,
      isConnecting: false,
    });

    render(<TaskProgress taskId="task-123" />);

    expect(screen.getByText('Status: COMPLETED')).toBeInTheDocument();
    expect(screen.getByText('Task finished successfully!')).toBeInTheDocument();
    expect(screen.getByText('100%')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '100');
  });

  test('displays failed status correctly', () => {
    mockUseTaskProgress.mockReturnValue({
      progressData: {
        task_id: 'task-123',
        status: 'FAILED',
        progress: 0.5, // Can be any value if failed mid-way
        message: 'Task failed: An unexpected error occurred.',
        error: 'Detailed error message.',
        start_time: Date.now() / 1000 - 30,
      },
      isConnected: true,
      isConnecting: false,
    });

    render(<TaskProgress taskId="task-123" />);

    expect(screen.getByText('Status: FAILED')).toBeInTheDocument();
    expect(screen.getByText('Task failed: An unexpected error occurred.')).toBeInTheDocument();
    expect(screen.getByText('Detailed error message.')).toBeInTheDocument();
    // Progress bar might still show the last known progress or 0
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  // 5. Visual representation of progress (e.g., progress bar).
  test('progress bar reflects percentage', () => {
    mockUseTaskProgress.mockReturnValue({
      progressData: {
        task_id: 'task-123',
        status: 'RUNNING',
        progress: 0.6,
        message: 'Working...',
        start_time: Date.now() / 1000 - 10,
      },
      isConnected: true,
      isConnecting: false,
    });

    render(<TaskProgress taskId="task-123" />);
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '60');
  });

  // 6. Time elapsed display
  test('displays time elapsed', async () => {
    jest.useFakeTimers();
    const startTime = Date.now() / 1000 - 125; // 2 minutes and 5 seconds ago

    mockUseTaskProgress.mockReturnValue({
      progressData: {
        task_id: 'task-123',
        status: 'RUNNING',
        progress: 0.5,
        message: 'Processing...',
        start_time: startTime,
      },
      isConnected: true,
      isConnecting: false,
    });

    render(<TaskProgress taskId="task-123" />);

    // Initial render might show 00:00 or very small value depending on exact timing
    // Advance timers to ensure calculation is stable
    act(() => {
      jest.advanceTimersByTime(1000); // Advance by 1 second
    });

    // Expecting it to show around 00:02:05 (2 minutes and 5 seconds)
    // Due to floating point and exact timing, we check for a range or substring
    await waitFor(() => {
      expect(screen.getByText(/Time Elapsed: 00:02:05/)).toBeInTheDocument();
    }, { timeout: 100 }); // Small timeout for waitFor

    jest.useRealTimers();
  });

  // 7. Integration with useTaskProgress hook (mocking the hook).
  test('uses data from useTaskProgress hook', () => {
    const mockData = {
      task_id: 'task-456',
      status: 'RUNNING',
      progress: 0.8,
      message: 'Finalizing report',
      start_time: Date.now() / 1000 - 300,
    };
    mockUseTaskProgress.mockReturnValue({
      progressData: mockData,
      isConnected: true,
      isConnecting: false,
    });

    render(<TaskProgress taskId="task-456" />);

    expect(mockUseTaskProgress).toHaveBeenCalledWith('task-456');
    expect(screen.getByText('Status: RUNNING')).toBeInTheDocument();
    expect(screen.getByText('Finalizing report')).toBeInTheDocument();
    expect(screen.getByText('80%')).toBeInTheDocument();
  });

  test('displays connection status', () => {
    mockUseTaskProgress.mockReturnValue({
      progressData: null,
      isConnected: false,
      isConnecting: true,
    });
    render(<TaskProgress taskId="task-123" />);
    expect(screen.getByText('Connecting to task progress...')).toBeInTheDocument();

    mockUseTaskProgress.mockReturnValue({
      progressData: null,
      isConnected: true,
      isConnecting: false,
    });
    render(<TaskProgress taskId="task-123" />);
    expect(screen.getByText('Connected')).toBeInTheDocument();
  });

  test('does not render if no task ID is provided', () => {
    const { container } = render(<TaskProgress taskId={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
