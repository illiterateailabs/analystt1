import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';

import AnalysisPanel from '../AnalysisPanel';
import * as api from '../../../lib/api';
import { useTaskProgress } from '../../../hooks/useTaskProgress';
import { useToast } from '../../../hooks/useToast';

// Mock the API functions
jest.mock('../../../lib/api', () => ({
  ...jest.requireActual('../../../lib/api'),
  analysisAPI: {
    fetchAnalysisResults: jest.fn(),
  },
}));

// Mock Next.js useRouter
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}));

// Mock useTaskProgress hook
jest.mock('../../../hooks/useTaskProgress', () => ({
  useTaskProgress: jest.fn(),
}));

// Mock useToast hook
jest.mock('../../../hooks/useToast', () => ({
  useToast: jest.fn(),
}));

// Mock sub-components
jest.mock('../../graph/GraphVisualization', () => ({
  __esModule: true,
  default: jest.fn(() => <div data-testid="mock-graph-visualization" />),
}));

jest.mock('../TaskProgress', () => ({
  __esModule: true,
  default: jest.fn(() => <div data-testid="mock-task-progress" />),
}));


const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false, // Disable retries for tests
    },
  },
});

const renderWithClient = (ui: React.ReactElement) => {
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
};

describe('AnalysisPanel', () => {
  const mockTaskId = 'test-task-id';
  const mockToast = jest.fn();
  const mockPush = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    queryClient.clear();
    (useRouter as jest.Mock).mockReturnValue({ push: mockPush });
    (useToast as jest.Mock).mockReturnValue({ toast: mockToast });

    // Default mock for useTaskProgress
    (useTaskProgress as jest.Mock).mockReturnValue({
      progressData: null,
      isConnected: false,
      isConnecting: false,
    });

    // Default successful mock for fetchAnalysisResults
    (api.analysisAPI.fetchAnalysisResults as jest.Mock).mockResolvedValue({
      task_id: mockTaskId,
      status: 'completed',
      title: 'Sample Analysis',
      executive_summary: 'This is a summary.',
      risk_score: 75,
      confidence: 0.9,
      detailed_findings: 'Detailed findings here.',
      graph_data: {
        nodes: [{ id: 'n1', label: 'Node 1' }],
        edges: [{ from: 'n1', to: 'n1', label: 'SELF_LOOP' }],
      },
      visualizations: [
        { filename: 'chart.png', content: 'base64image', type: 'image/png' },
      ],
      recommendations: ['Recommendation 1', 'Recommendation 2'],
      code_generated: 'print("hello")',
      execution_details: {},
    });
  });

  // 1. Rendering with initial analysis data
  test('renders AnalysisPanel with initial data', async () => {
    renderWithClient(<AnalysisPanel taskId={mockTaskId} />);

    await waitFor(() => {
      expect(screen.getByText('Sample Analysis')).toBeInTheDocument();
      expect(screen.getByText('This is a summary.')).toBeInTheDocument();
      expect(screen.getByText('Risk Score: 75')).toBeInTheDocument();
      expect(screen.getByText('Confidence: 90%')).toBeInTheDocument();
      expect(screen.getByText('Detailed findings here.')).toBeInTheDocument();
      expect(screen.getByText('Recommendation 1')).toBeInTheDocument();
      expect(screen.getByText('Recommendation 2')).toBeInTheDocument();
    });
  });

  // 2. Displaying analysis results, including graph data and visualizations
  test('displays graph visualization and code when tabs are selected', async () => {
    renderWithClient(<AnalysisPanel taskId={mockTaskId} />);

    await waitFor(() => {
      expect(screen.getByText('Sample Analysis')).toBeInTheDocument();
    });

    // Switch to Graph tab
    await act(async () => {
      userEvent.click(screen.getByRole('tab', { name: /graph/i }));
    });
    expect(screen.getByTestId('mock-graph-visualization')).toBeInTheDocument();

    // Switch to Code tab
    await act(async () => {
      userEvent.click(screen.getByRole('tab', { name: /code/i }));
    });
    expect(screen.getByText('print("hello")')).toBeInTheDocument();

    // Switch to Visualizations tab
    await act(async () => {
      userEvent.click(screen.getByRole('tab', { name: /visualizations/i }));
    });
    expect(screen.getByAltText('chart.png')).toBeInTheDocument();
  });

  // 3. Handling different analysis statuses (loading, completed, failed)
  test('shows loading state when analysis is in progress', async () => {
    (api.analysisAPI.fetchAnalysisResults as jest.Mock).mockResolvedValueOnce({
      task_id: mockTaskId,
      status: 'running',
      title: 'Analysis in Progress',
    });
    (useTaskProgress as jest.Mock).mockReturnValue({
      progressData: {
        task_id: mockTaskId,
        status: 'RUNNING',
        progress: 0.5,
        message: 'Processing data',
      },
      isConnected: true,
      isConnecting: false,
    });

    renderWithClient(<AnalysisPanel taskId={mockTaskId} />);

    await waitFor(() => {
      expect(screen.getByText('Analysis in Progress')).toBeInTheDocument();
      expect(screen.getByTestId('mock-task-progress')).toBeInTheDocument();
      expect(screen.getByRole('progressbar')).toBeInTheDocument(); // CircularProgress
    });
  });

  test('shows error state when analysis fails', async () => {
    (api.analysisAPI.fetchAnalysisResults as jest.Mock).mockResolvedValueOnce({
      task_id: mockTaskId,
      status: 'failed',
      error: 'Analysis failed due to an internal error.',
    });

    renderWithClient(<AnalysisPanel taskId={mockTaskId} />);

    await waitFor(() => {
      expect(screen.getByText('Analysis Failed')).toBeInTheDocument();
      expect(screen.getByText('Analysis failed due to an internal error.')).toBeInTheDocument();
      expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    });
  });

  // 4. User interactions like selecting different analysis views or parameters
  test('allows switching between tabs', async () => {
    renderWithClient(<AnalysisPanel taskId={mockTaskId} />);

    await waitFor(() => {
      expect(screen.getByText('Sample Analysis')).toBeInTheDocument();
    });

    // Initial tab is Overview
    expect(screen.getByRole('tab', { name: /overview/i })).toHaveAttribute('aria-selected', 'true');

    // Switch to Graph tab
    await act(async () => {
      userEvent.click(screen.getByRole('tab', { name: /graph/i }));
    });
    expect(screen.getByRole('tab', { name: /graph/i })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByTestId('mock-graph-visualization')).toBeInTheDocument();

    // Switch to Code tab
    await act(async () => {
      userEvent.click(screen.getByRole('tab', { name: /code/i }));
    });
    expect(screen.getByRole('tab', { name: /code/i })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByText('print("hello")')).toBeInTheDocument();
  });

  // 5. Integration with API calls for fetching analysis results
  test('calls fetchAnalysisResults with correct task ID', async () => {
    renderWithClient(<AnalysisPanel taskId={mockTaskId} />);

    await waitFor(() => {
      expect(api.analysisAPI.fetchAnalysisResults).toHaveBeenCalledWith(mockTaskId);
    });
  });

  // 6. Error handling for API failures
  test('displays error message when fetchAnalysisResults API fails', async () => {
    (api.analysisAPI.fetchAnalysisResults as jest.Mock).mockRejectedValue(new Error('Network error'));

    renderWithClient(<AnalysisPanel taskId={mockTaskId} />);

    await waitFor(() => {
      expect(screen.getByText('Error loading analysis results')).toBeInTheDocument();
      expect(screen.getByText('Failed to load analysis results. Please try again.')).toBeInTheDocument();
    });
  });

  // 7. Rendering of sub-components like GraphVisualization and TaskProgress
  test('renders GraphVisualization when graph data is available and tab is selected', async () => {
    renderWithClient(<AnalysisPanel taskId={mockTaskId} />);

    await waitFor(() => {
      expect(screen.getByText('Sample Analysis')).toBeInTheDocument();
    });

    await act(async () => {
      userEvent.click(screen.getByRole('tab', { name: /graph/i }));
    });

    expect(screen.getByTestId('mock-graph-visualization')).toBeInTheDocument();
  });

  test('renders TaskProgress when analysis is running', async () => {
    (api.analysisAPI.fetchAnalysisResults as jest.Mock).mockResolvedValueOnce({
      task_id: mockTaskId,
      status: 'running',
      title: 'Analysis in Progress',
    });
    (useTaskProgress as jest.Mock).mockReturnValue({
      progressData: {
        task_id: mockTaskId,
        status: 'RUNNING',
        progress: 0.5,
        message: 'Processing data',
      },
      isConnected: true,
      isConnecting: false,
    });

    renderWithClient(<AnalysisPanel taskId={mockTaskId} />);

    await waitFor(() => {
      expect(screen.getByTestId('mock-task-progress')).toBeInTheDocument();
    });
  });
});
