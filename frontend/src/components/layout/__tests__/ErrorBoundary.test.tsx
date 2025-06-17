import React, { ErrorInfo } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import ErrorBoundary from '../ErrorBoundary';

// Mock console.error to prevent test output pollution
const originalConsoleError = console.error;
beforeAll(() => {
  console.error = jest.fn();
});

afterAll(() => {
  console.error = originalConsoleError;
});

// Reset console.error mock between tests
beforeEach(() => {
  (console.error as jest.Mock).mockClear();
});

// Component that throws an error when shouldThrow is true
const ErrorThrowingComponent = ({ shouldThrow = false, message = 'Test error' }) => {
  if (shouldThrow) {
    throw new Error(message);
  }
  return <div data-testid="normal-component">Normal Component Content</div>;
};

describe('ErrorBoundary', () => {
  test('renders children when no error occurs', () => {
    render(
      <ErrorBoundary>
        <ErrorThrowingComponent shouldThrow={false} />
      </ErrorBoundary>
    );

    expect(screen.getByTestId('normal-component')).toBeInTheDocument();
    expect(screen.getByText('Normal Component Content')).toBeInTheDocument();
  });

  test('displays default fallback UI when error occurs', () => {
    // Suppress React error boundary warning in test output
    const spy = jest.spyOn(console, 'error');
    spy.mockImplementation(() => {});

    render(
      <ErrorBoundary>
        <ErrorThrowingComponent shouldThrow={true} message="Test error message" />
      </ErrorBoundary>
    );

    // Verify fallback UI is shown
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();

    // In development mode, error details should be visible
    if (process.env.NODE_ENV === 'development') {
      expect(screen.getByText('Error Details (visible in development only):')).toBeInTheDocument();
      expect(screen.getByText('Error: Test error message')).toBeInTheDocument();
    }

    spy.mockRestore();
  });

  test('displays custom fallback UI when provided', () => {
    const customFallback = <div data-testid="custom-fallback">Custom Error UI</div>;
    
    render(
      <ErrorBoundary fallback={customFallback}>
        <ErrorThrowingComponent shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByTestId('custom-fallback')).toBeInTheDocument();
    expect(screen.getByText('Custom Error UI')).toBeInTheDocument();
  });

  test('calls onError callback when error occurs', () => {
    const handleError = jest.fn();
    
    render(
      <ErrorBoundary onError={handleError}>
        <ErrorThrowingComponent shouldThrow={true} message="Callback test error" />
      </ErrorBoundary>
    );

    // Verify onError was called with the error
    expect(handleError).toHaveBeenCalledTimes(1);
    expect(handleError.mock.calls[0][0].message).toBe('Callback test error');
    expect(handleError.mock.calls[0][1]).toBeDefined(); // ErrorInfo object
  });

  test('resets error state when Try again button is clicked', () => {
    // We need to mock React's setState since we're testing a class component
    const ErrorResetComponent = () => {
      const [shouldThrow, setShouldThrow] = React.useState(true);
      
      // Reset the error state after the component has thrown once
      React.useEffect(() => {
        if (shouldThrow) {
          setTimeout(() => setShouldThrow(false), 0);
        }
      }, [shouldThrow]);
      
      if (shouldThrow) {
        throw new Error('Initial error');
      }
      
      return <div data-testid="reset-success">Component Reset Successfully</div>;
    };

    render(
      <ErrorBoundary>
        <ErrorResetComponent />
      </ErrorBoundary>
    );

    // First we should see the error UI
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    
    // Click the reset button
    fireEvent.click(screen.getByRole('button', { name: /try again/i }));
    
    // After reset, we should see the normal component
    expect(screen.getByTestId('reset-success')).toBeInTheDocument();
    expect(screen.getByText('Component Reset Successfully')).toBeInTheDocument();
  });

  test('resets when resetKey prop changes', () => {
    // Component that renders ErrorBoundary with changing resetKey
    const ResetKeyWrapper = () => {
      const [resetKey, setResetKey] = React.useState(0);
      const [shouldThrow, setShouldThrow] = React.useState(true);
      
      const handleReset = () => {
        setShouldThrow(false);
        setResetKey(prev => prev + 1);
      };
      
      return (
        <div>
          <ErrorBoundary resetKey={resetKey}>
            <ErrorThrowingComponent shouldThrow={shouldThrow} />
          </ErrorBoundary>
          <button onClick={handleReset} data-testid="external-reset">External Reset</button>
        </div>
      );
    };

    render(<ResetKeyWrapper />);
    
    // First we should see the error UI
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    
    // Click the external reset button
    fireEvent.click(screen.getByTestId('external-reset'));
    
    // After reset, we should see the normal component
    expect(screen.getByTestId('normal-component')).toBeInTheDocument();
    expect(screen.getByText('Normal Component Content')).toBeInTheDocument();
  });

  test('handles nested error boundaries correctly', () => {
    render(
      <ErrorBoundary fallback={<div data-testid="outer-fallback">Outer Error</div>}>
        <div>Outer Content</div>
        <ErrorBoundary fallback={<div data-testid="inner-fallback">Inner Error</div>}>
          <ErrorThrowingComponent shouldThrow={true} />
        </ErrorBoundary>
      </ErrorBoundary>
    );

    // The inner error boundary should catch the error
    expect(screen.getByTestId('inner-fallback')).toBeInTheDocument();
    expect(screen.getByText('Inner Error')).toBeInTheDocument();
    
    // The outer content should still be visible
    expect(screen.getByText('Outer Content')).toBeInTheDocument();
    
    // The outer fallback should not be shown
    expect(screen.queryByTestId('outer-fallback')).not.toBeInTheDocument();
  });

  test('integrates with Sentry if available', () => {
    // Mock window.Sentry
    const mockCaptureException = jest.fn();
    const originalWindow = { ...window };
    
    // @ts-ignore - Adding Sentry to window for testing
    window.Sentry = { 
      captureException: mockCaptureException 
    };
    
    render(
      <ErrorBoundary>
        <ErrorThrowingComponent shouldThrow={true} message="Sentry test error" />
      </ErrorBoundary>
    );

    // Verify Sentry.captureException was called
    expect(mockCaptureException).toHaveBeenCalledTimes(1);
    expect(mockCaptureException.mock.calls[0][0].message).toBe('Sentry test error');
    
    // Restore original window
    window = originalWindow;
  });
});
