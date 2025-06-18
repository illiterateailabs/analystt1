import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import ComplianceReview from '../ComplianceReview';
import * as api from '../../../lib/api';
import { useToast } from '../../../hooks/useToast';
import { useAuth } from '../../../hooks/useAuth';

// Mock the API functions
jest.mock('../../../lib/api', () => ({
  ...jest.requireActual('../../../lib/api'),
  crewAPI: {
    resumeTask: jest.fn(),
  },
}));

// Mock useToast hook
jest.mock('../../../hooks/useToast', () => ({
  useToast: jest.fn(),
}));

// Mock useAuth hook
jest.mock('../../../hooks/useAuth', () => ({
  useAuth: jest.fn(),
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

describe('ComplianceReview', () => {
  const mockTaskId = 'task-123';
  const mockToast = jest.fn();
  const mockOnReviewComplete = jest.fn();
  const mockUser = { username: 'testreviewer' };

  beforeEach(() => {
    jest.clearAllMocks();
    queryClient.clear();
    (useToast as jest.Mock).mockReturnValue({ toast: mockToast });
    (useAuth as jest.Mock).mockReturnValue({ user: mockUser });

    // Default successful mock for resumeTask
    (api.crewAPI.resumeTask as jest.Mock).mockResolvedValue({
      success: true,
      message: 'Review submitted successfully',
    });
  });

  // 1. Rendering with initial review data (if applicable).
  test('renders the compliance review form correctly', () => {
    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );

    expect(screen.getByLabelText(/Findings/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Risk Level/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Regulatory Implications/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Comments/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Approve/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Reject/i })).toBeInTheDocument();
  });

  // 2. Form input rendering and validation.
  test('allows input in form fields', async () => {
    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );

    const findingsInput = screen.getByLabelText(/Findings/i);
    const riskLevelSelect = screen.getByLabelText(/Risk Level/i);
    const implicationsInput = screen.getByLabelText(/Regulatory Implications/i);
    const commentsInput = screen.getByLabelText(/Comments/i);

    await userEvent.type(findingsInput, 'Some important findings.');
    await userEvent.selectOptions(riskLevelSelect, 'Medium');
    await userEvent.type(implicationsInput, 'AML, KYC');
    await userEvent.type(commentsInput, 'Looks suspicious.');

    expect(findingsInput).toHaveValue('Some important findings.');
    expect(riskLevelSelect).toHaveValue('Medium');
    expect(implicationsInput).toHaveValue('AML, KYC');
    expect(commentsInput).toHaveValue('Looks suspicious.');
  });

  test('shows validation error if findings are empty on submit', async () => {
    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );
    const approveButton = screen.getByRole('button', { name: /Approve/i });

    await userEvent.click(approveButton);

    expect(screen.getByText('Findings are required.')).toBeInTheDocument();
    expect(api.crewAPI.resumeTask).not.toHaveBeenCalled();
  });

  // 3. Submitting the review and handling API call (mock `crewAPI.resumeTask`).
  // 6. Handling different review statuses (approve, reject).
  test('submits review with "Approve" status and calls API', async () => {
    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );

    const findingsInput = screen.getByLabelText(/Findings/i);
    const riskLevelSelect = screen.getByLabelText(/Risk Level/i);
    const approveButton = screen.getByRole('button', { name: /Approve/i });

    await userEvent.type(findingsInput, 'All good.');
    await userEvent.selectOptions(riskLevelSelect, 'Low');
    await userEvent.click(approveButton);

    await waitFor(() => {
      expect(api.crewAPI.resumeTask).toHaveBeenCalledWith(mockTaskId, {
        status: 'approved',
        reviewer: mockUser.username,
        findings: 'All good.',
        risk_level: 'Low',
        regulatory_implications: [], // Default if not filled
        comments: '', // Default if not filled
      });
    });
  });

  test('submits review with "Reject" status and calls API', async () => {
    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );

    const findingsInput = screen.getByLabelText(/Findings/i);
    const riskLevelSelect = screen.getByLabelText(/Risk Level/i);
    const implicationsInput = screen.getByLabelText(/Regulatory Implications/i);
    const commentsInput = screen.getByLabelText(/Comments/i);
    const rejectButton = screen.getByRole('button', { name: /Reject/i });

    await userEvent.type(findingsInput, 'Serious issues found.');
    await userEvent.selectOptions(riskLevelSelect, 'High');
    await userEvent.type(implicationsInput, 'Sanctions violation');
    await userEvent.type(commentsInput, 'Needs immediate escalation.');
    await userEvent.click(rejectButton);

    await waitFor(() => {
      expect(api.crewAPI.resumeTask).toHaveBeenCalledWith(mockTaskId, {
        status: 'rejected',
        reviewer: mockUser.username,
        findings: 'Serious issues found.',
        risk_level: 'High',
        regulatory_implications: ['Sanctions violation'],
        comments: 'Needs immediate escalation.',
      });
    });
  });

  // 4. Displaying loading state during submission.
  test('displays loading state during submission', async () => {
    (api.crewAPI.resumeTask as jest.Mock).mockImplementation(
      () => new Promise(resolve => setTimeout(() => resolve({ success: true }), 100))
    );

    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );
    const findingsInput = screen.getByLabelText(/Findings/i);
    const approveButton = screen.getByRole('button', { name: /Approve/i });

    await userEvent.type(findingsInput, 'Submitting...');
    await userEvent.click(approveButton);

    expect(approveButton).toBeDisabled();
    expect(screen.getByRole('button', { name: /Reject/i })).toBeDisabled();
    expect(screen.getByRole('progressbar')).toBeInTheDocument();

    await waitFor(() => {
      expect(approveButton).not.toBeDisabled();
    });
  });

  // 5. Displaying success/error messages after submission.
  test('displays success message and calls onReviewComplete on successful submission', async () => {
    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );
    const findingsInput = screen.getByLabelText(/Findings/i);
    const approveButton = screen.getByRole('button', { name: /Approve/i });

    await userEvent.type(findingsInput, 'Success test.');
    await userEvent.click(approveButton);

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({
        description: 'Review submitted successfully.',
        variant: 'success',
      }));
      expect(mockOnReviewComplete).toHaveBeenCalledTimes(1);
    });
  });

  test('displays error message on failed submission', async () => {
    (api.crewAPI.resumeTask as jest.Mock).mockRejectedValue(new Error('API submission error'));
    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );
    const findingsInput = screen.getByLabelText(/Findings/i);
    const approveButton = screen.getByRole('button', { name: /Approve/i });

    await userEvent.type(findingsInput, 'Error submission test.');
    await userEvent.click(approveButton);

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({
        description: 'Failed to submit review. Please try again.',
        variant: 'destructive',
      }));
      expect(mockOnReviewComplete).not.toHaveBeenCalled();
    });
  });

  // 7. Interaction with any parent components or context if applicable.
  // (Covered by onReviewComplete mock and useAuth mock for reviewer name)
  test('uses username from useAuth as reviewer', async () => {
    (useAuth as jest.Mock).mockReturnValue({ user: { username: 'customReviewer' } });
    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );
    const findingsInput = screen.getByLabelText(/Findings/i);
    const approveButton = screen.getByRole('button', { name: /Approve/i });

    await userEvent.type(findingsInput, 'Reviewer test.');
    await userEvent.click(approveButton);

    await waitFor(() => {
      expect(api.crewAPI.resumeTask).toHaveBeenCalledWith(mockTaskId, expect.objectContaining({
        reviewer: 'customReviewer',
      }));
    });
  });

  test('handles case where useAuth returns no user', async () => {
    (useAuth as jest.Mock).mockReturnValue({ user: null });
    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );
    const findingsInput = screen.getByLabelText(/Findings/i);
    const approveButton = screen.getByRole('button', { name: /Approve/i });

    await userEvent.type(findingsInput, 'No user test.');
    await userEvent.click(approveButton);

    await waitFor(() => {
      expect(api.crewAPI.resumeTask).toHaveBeenCalledWith(mockTaskId, expect.objectContaining({
        reviewer: 'Unknown Reviewer', // Default reviewer name
      }));
    });
  });

  test('regulatory implications are split by comma', async () => {
    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );
    const findingsInput = screen.getByLabelText(/Findings/i);
    const implicationsInput = screen.getByLabelText(/Regulatory Implications/i);
    const approveButton = screen.getByRole('button', { name: /Approve/i });

    await userEvent.type(findingsInput, 'Implications test.');
    await userEvent.type(implicationsInput, 'GDPR, CCPA, HIPAA');
    await userEvent.click(approveButton);

    await waitFor(() => {
      expect(api.crewAPI.resumeTask).toHaveBeenCalledWith(mockTaskId, expect.objectContaining({
        regulatory_implications: ['GDPR', 'CCPA', 'HIPAA'],
      }));
    });
  });

  test('regulatory implications handles empty string', async () => {
    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );
    const findingsInput = screen.getByLabelText(/Findings/i);
    const implicationsInput = screen.getByLabelText(/Regulatory Implications/i);
    const approveButton = screen.getByRole('button', { name: /Approve/i });

    await userEvent.type(findingsInput, 'Empty implications test.');
    await userEvent.type(implicationsInput, ''); // Empty string
    await userEvent.click(approveButton);

    await waitFor(() => {
      expect(api.crewAPI.resumeTask).toHaveBeenCalledWith(mockTaskId, expect.objectContaining({
        regulatory_implications: [],
      }));
    });
  });

  test('regulatory implications handles string with only commas/spaces', async () => {
    renderWithClient(
      <ComplianceReview taskId={mockTaskId} onReviewComplete={mockOnReviewComplete} />
    );
    const findingsInput = screen.getByLabelText(/Findings/i);
    const implicationsInput = screen.getByLabelText(/Regulatory Implications/i);
    const approveButton = screen.getByRole('button', { name: /Approve/i });

    await userEvent.type(findingsInput, 'Spaced implications test.');
    await userEvent.type(implicationsInput, ' , ,,  , '); // String with only commas and spaces
    await userEvent.click(approveButton);

    await waitFor(() => {
      expect(api.crewAPI.resumeTask).toHaveBeenCalledWith(mockTaskId, expect.objectContaining({
        regulatory_implications: [],
      }));
    });
  });
});
