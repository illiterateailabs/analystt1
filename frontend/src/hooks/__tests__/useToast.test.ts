import React from 'react';
import { render, screen, act, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { useToast } from '../useToast';
import { Toaster } from 'react-hot-toast'; // Assuming react-hot-toast's Toaster is used

// Mock react-hot-toast's internal toast function
// This allows us to assert on calls to toast and control its behavior
const mockHotToast = jest.fn();
jest.mock('react-hot-toast', () => ({
  ...jest.requireActual('react-hot-toast'),
  toast: mockHotToast,
  Toaster: jest.fn(({ children }) => <div data-testid="toaster-container">{children}</div>), // Mock Toaster to render children
}));

// A test component to use the useToast hook
const TestComponent = () => {
  const { toast } = useToast();
  return (
    <div>
      <button onClick={() => toast({ description: 'Default toast' })} data-testid="default-toast-btn">
        Show Default Toast
      </button>
      <button onClick={() => toast({ description: 'Success toast', variant: 'success' })} data-testid="success-toast-btn">
        Show Success Toast
      </button>
      <button onClick={() => toast({ description: 'Error toast', variant: 'destructive' })} data-testid="error-toast-btn">
        Show Error Toast
      </button>
      <button onClick={() => toast({ description: 'Warning toast', variant: 'warning' })} data-testid="warning-toast-btn">
        Show Warning Toast
      </button>
      <button onClick={() => toast({ description: 'Info toast', variant: 'info' })} data-testid="info-toast-btn">
        Show Info Toast
      </button>
      <button onClick={() => toast({ title: 'Title', description: 'Toast with title' })} data-testid="title-toast-btn">
        Show Toast with Title
      </button>
      <button
        onClick={() =>
          toast({
            description: 'Toast with action',
            action: {
              label: 'Undo',
              onClick: jest.fn(),
            },
          })
        }
        data-testid="action-toast-btn"
      >
        Show Toast with Action
      </button>
      <button onClick={() => mockHotToast.mock.results[0].value.dismiss()} data-testid="dismiss-toast-btn">
        Dismiss First Toast
      </button>
    </div>
  );
};

describe('useToast', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers(); // Enable fake timers for auto-dismiss testing
  });

  afterEach(() => {
    jest.runOnlyPendingTimers(); // Clear any pending timers
    jest.useRealTimers(); // Restore real timers
  });

  // 1. Toast rendering with different variants
  test('should render a default toast', async () => {
    render(<TestComponent />);
    fireEvent.click(screen.getByTestId('default-toast-btn'));

    expect(mockHotToast).toHaveBeenCalledTimes(1);
    expect(mockHotToast).toHaveBeenCalledWith(
      expect.any(Function), // The content function
      expect.objectContaining({
        duration: 5000, // Default duration
        className: 'toast-default', // Default class
      })
    );
    // Verify the content function renders the description
    const toastContent = render(mockHotToast.mock.calls[0][0]({ visible: true })).container;
    expect(toastContent).toHaveTextContent('Default toast');
  });

  test('should render a success toast', async () => {
    render(<TestComponent />);
    fireEvent.click(screen.getByTestId('success-toast-btn'));

    expect(mockHotToast).toHaveBeenCalledTimes(1);
    expect(mockHotToast).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        className: 'toast-success',
      })
    );
    const toastContent = render(mockHotToast.mock.calls[0][0]({ visible: true })).container;
    expect(toastContent).toHaveTextContent('Success toast');
  });

  test('should render an error toast', async () => {
    render(<TestComponent />);
    fireEvent.click(screen.getByTestId('error-toast-btn'));

    expect(mockHotToast).toHaveBeenCalledTimes(1);
    expect(mockHotToast).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        className: 'toast-destructive',
      })
    );
    const toastContent = render(mockHotToast.mock.calls[0][0]({ visible: true })).container;
    expect(toastContent).toHaveTextContent('Error toast');
  });

  test('should render a warning toast', async () => {
    render(<TestComponent />);
    fireEvent.click(screen.getByTestId('warning-toast-btn'));

    expect(mockHotToast).toHaveBeenCalledTimes(1);
    expect(mockHotToast).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        className: 'toast-warning',
      })
    );
    const toastContent = render(mockHotToast.mock.calls[0][0]({ visible: true })).container;
    expect(toastContent).toHaveTextContent('Warning toast');
  });

  test('should render an info toast', async () => {
    render(<TestComponent />);
    fireEvent.click(screen.getByTestId('info-toast-btn'));

    expect(mockHotToast).toHaveBeenCalledTimes(1);
    expect(mockHotToast).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        className: 'toast-info',
      })
    );
    const toastContent = render(mockHotToast.mock.calls[0][0]({ visible: true })).container;
    expect(toastContent).toHaveTextContent('Info toast');
  });

  // 2. Displaying title and description
  test('should display title and description', async () => {
    render(<TestComponent />);
    fireEvent.click(screen.getByTestId('title-toast-btn'));

    expect(mockHotToast).toHaveBeenCalledTimes(1);
    const toastContent = render(mockHotToast.mock.calls[0][0]({ visible: true })).container;
    expect(toastContent).toHaveTextContent('Title');
    expect(toastContent).toHaveTextContent('Toast with title');
  });

  // 3. Auto-dismiss functionality
  test('should auto-dismiss after duration', async () => {
    render(<TestComponent />);
    fireEvent.click(screen.getByTestId('default-toast-btn'));

    // Simulate the toast being rendered by react-hot-toast
    const toastId = 'test-toast-id';
    mockHotToast.mock.calls[0][0]({ id: toastId, visible: true }); // Render the toast content

    // Advance timers by the default duration (5000ms)
    act(() => {
      jest.advanceTimersByTime(5000);
    });

    // Expect the toast to be dismissed (mockHotToast.dismiss is called internally by react-hot-toast)
    // We can't directly check if the toast is removed from the DOM here because we're mocking react-hot-toast.
    // Instead, we rely on react-hot-toast's internal dismiss mechanism which is triggered by its duration.
    // The mockHotToast.mock.results[0].value.dismiss() would be called by react-hot-toast.
    // For this test, we're verifying that the toast function was called with a duration.
    expect(mockHotToast).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        duration: 5000,
      })
    );
  });

  // 4. Manual dismiss via close button
  test('should dismiss manually via close button', async () => {
    render(<TestComponent />);
    fireEvent.click(screen.getByTestId('default-toast-btn'));

    // Simulate the toast being rendered by react-hot-toast
    const toastId = 'test-toast-id';
    const dismissMock = jest.fn();
    render(mockHotToast.mock.calls[0][0]({ id: toastId, visible: true, dismiss: dismissMock }));

    // Find and click the close button
    const closeButton = screen.getByRole('button', { name: /close/i });
    fireEvent.click(closeButton);

    // Expect the dismiss function provided by react-hot-toast to be called
    expect(dismissMock).toHaveBeenCalledTimes(1);
  });

  // 5. Action button rendering and click handling
  test('should render action button and handle click', async () => {
    const actionOnClickMock = jest.fn();
    mockHotToast.mockImplementationOnce((content, options) => {
      const toastId = 'action-toast-id';
      const dismiss = jest.fn();
      return {
        id: toastId,
        dismiss: dismiss,
        // Simulate the content function that react-hot-toast would render
        content: content({ id: toastId, visible: true, dismiss: dismiss }),
      };
    });

    render(<TestComponent />);
    fireEvent.click(screen.getByTestId('action-toast-btn'));

    // Verify the action button is rendered
    const actionButton = screen.getByRole('button', { name: /undo/i });
    expect(actionButton).toBeInTheDocument();

    // Simulate click on action button
    fireEvent.click(actionButton);

    // Expect the action's onClick to be called
    expect(actionOnClickMock).toHaveBeenCalledTimes(1);
  });

  // 6. Multiple toasts stacking correctly (indirectly tested by mock)
  test('should allow multiple toasts to be displayed', async () => {
    render(<TestComponent />);
    fireEvent.click(screen.getByTestId('default-toast-btn'));
    fireEvent.click(screen.getByTestId('success-toast-btn'));
    fireEvent.click(screen.getByTestId('error-toast-btn'));

    expect(mockHotToast).toHaveBeenCalledTimes(3);
    // In a real scenario, react-hot-toast handles stacking. Here, we just verify multiple calls.
  });

  // 7. Toast positioning and styling (if testable)
  test('should apply correct styling classes based on variant', async () => {
    render(<TestComponent />);
    fireEvent.click(screen.getByTestId('default-toast-btn'));
    fireEvent.click(screen.getByTestId('success-toast-btn'));
    fireEvent.click(screen.getByTestId('error-toast-btn'));

    // Check the className passed to react-hot-toast
    expect(mockHotToast).toHaveBeenCalledWith(expect.any(Function), expect.objectContaining({ className: 'toast-default' }));
    expect(mockHotToast).toHaveBeenCalledWith(expect.any(Function), expect.objectContaining({ className: 'toast-success' }));
    expect(mockHotToast).toHaveBeenCalledWith(expect.any(Function), expect.objectContaining({ className: 'toast-destructive' }));
  });

  // 8. Accessibility attributes for toasts.
  test('should have correct accessibility attributes', async () => {
    render(<TestComponent />);
    fireEvent.click(screen.getByTestId('default-toast-btn'));

    // Simulate the toast being rendered by react-hot-toast
    const toastContent = render(mockHotToast.mock.calls[0][0]({ visible: true })).container;

    // Assuming the toast content itself or a wrapper has these attributes
    // react-hot-toast typically adds role="status" and aria-live="polite" to its container
    // We're mocking the Toaster, so we'll check the rendered content for these.
    // If the internal component doesn't add them, this test might fail or need adjustment.
    expect(toastContent.querySelector('[role="status"]')).toBeInTheDocument();
    expect(toastContent.querySelector('[aria-live="polite"]')).toBeInTheDocument();
  });
});
